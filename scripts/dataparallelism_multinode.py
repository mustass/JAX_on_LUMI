import functools
from typing import Callable
import time
import jax
import numpy as np

jax.distributed.initialize() 
from jax import random, numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import linen as nn
from flax.training import train_state

import optax # Optax for common losses and optimizers.

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloader(batch_size=4096):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='/scratch/project_465001020/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True )
    return trainloader


class CNN(nn.Module):
    num_classes: int
    dense_init: Callable = nn.initializers.xavier_normal()
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (3, 3), padding='SAME', kernel_init=self.dense_init)(x))
        x = nn.max_pool(x, (2, 2))
        x = nn.relu(nn.Conv(64, (3, 3), padding='SAME', kernel_init=self.dense_init)(x))
        x = nn.max_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.relu(nn.Dense(128,kernel_init=self.dense_init,
                            use_bias=False
                 )(x))
        x = nn.Dense(self.num_classes,kernel_init=self.dense_init,
                            use_bias=False,
                 )(x)
        return x


def to_jax_array(np_array):
    return jnp.array(np_array)

def init_fn(k, x, model, optimizer):
    variables = model.init(k, x) # Initialize the model.
    state = train_state.TrainState.create( # Create a `TrainState`.
      apply_fn=model.apply,
      params=variables['params'],
      tx=optimizer)
    return state


def get_state_sharding(init_fn,key,x,model,optimizer, mesh):

  abstract_variables = jax.eval_shape(
      functools.partial(init_fn, model=model, optimizer=optimizer), key, x)

  state_sharding = nn.get_sharding(abstract_variables, mesh)
  return state_sharding


"""Then, create a compiled inference step. Note that the output is also sharded along `(data, None)`."""

def model_predict(x,y, x_sharding, state_sharding,mesh,state):
  @functools.partial(jax.jit, in_shardings=(state_sharding, x_sharding),
                    out_shardings=x_sharding)
  def apply_fn(state, x):
    return state.apply_fn({'params': state.params}, x)

  with mesh:
    preds = apply_fn(state, x)
  jax.debug.visualize_array_sharding(preds)
  print(f"Preds: {jnp.argmax(jax.nn.softmax(preds), axis=-1)}\n vs actual:\n {y}")

def ggn_vp(loader, state,mesh_sharding,x_sharding, global_shape_x):
  vec_sharding = mesh_sharding(PartitionSpec(None)) # the params_vec is replicated on each device

  params_vec, unflatten_fn = jax.flatten_util.ravel_pytree(state.params)

  @functools.partial(jax.jit, in_shardings=(vec_sharding, x_sharding),
                    out_shardings=x_sharding)
  def apply_fn_vec(params_vec, x):
    return state.apply_fn({'params': unflatten_fn(params_vec)}, x)

  vec_shape = (len(params_vec),)
  vec = jnp.ones(vec_shape)
  vec_arrays = [
      jax.device_put(vec, d)
      for d, _ in x_sharding.addressable_devices_indices_map(global_shape_x).items()]
  
  vec = jax.make_array_from_single_device_arrays((len(params_vec)*len(vec_arrays),), vec_sharding, vec_arrays)


  ggn_vp_result = 0

  for (x,y) in loader:
    x = to_jax_array(x)
    y= to_jax_array(y)
    x = jnp.transpose(x,(0,2,3,1))
    x_arrays = [
      jax.device_put(x[index], d)
      for d, index in x_sharding.addressable_devices_indices_map(global_shape_x).items()]
    x = jax.make_array_from_single_device_arrays(global_shape_x, x_sharding, x_arrays)
    model_on_data = lambda p: apply_fn_vec(p, x)
    _, jvp = jax.jvp(model_on_data, (params_vec,), (vec,))
    _, vjp_fn = jax.vjp(model_on_data, params_vec)
    ggn_vp_result += vjp_fn(jvp)[0]
  
  print("DONE")

def run(EPOCHS):
  print(f"Get loader")
  loader = get_dataloader()

  mesh = Mesh(np.array(jax.devices()).reshape(jax.device_count()), ('data'))
  sharding = jax.sharding.NamedSharding(mesh, P('data'))

  def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)
  
  # MLP hyperparameters.
  BATCH, DIM, CHANNELS, NUM_CLASS = 4096, 32, 3, 10

  global_shape_x = (BATCH,DIM,DIM,CHANNELS)
  global_shape_y = (BATCH,)
  # Create fake inputs.
  x = jnp.ones(global_shape_x)
  y = jnp.ones(global_shape_y)
  # Initialize a PRNG key.
  k = random.key(0)

  # Create an Optax optimizer.
  optimizer = optax.adam(learning_rate=0.001)
  # Instantiate the model.
  model = CNN(NUM_CLASS)

  print(f"x shape is {x.shape}")
  x_arrays = [
  jax.device_put(x[index], d)
      for d, index in sharding.addressable_devices_indices_map(global_shape_x).items()]
  y_arrays = [
  jax.device_put(y[index], d)
      for d, index in sharding.addressable_devices_indices_map(global_shape_y).items()]
  
  x = jax.make_array_from_single_device_arrays(global_shape_x, sharding, x_arrays)
  y = jax.make_array_from_single_device_arrays(global_shape_y, sharding, y_arrays)

  state_sharding = get_state_sharding(init_fn,k,x,model,optimizer,mesh)

  jit_init_fn = jax.jit(init_fn, static_argnums=(2, 3),
                      in_shardings=(mesh_sharding(()), sharding),  # PRNG key and x
                      out_shardings=state_sharding)

  state = jit_init_fn(k, x, model, optimizer)

  @functools.partial(jax.jit, in_shardings=(state_sharding, sharding, sharding),
                   out_shardings=state_sharding)
  def train_step(state, x, y):
    # A fake loss function.
    def loss_unrolled(params):
      logits = model.apply({'params': params}, x)
      return jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(y, 10)))

    grad_fn = jax.grad(loss_unrolled)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


  for e in range(EPOCHS):
    print(f"Running epoch {e}")
    print(f"Batches: {len(loader)}")
    t0 = time.time()
    for i, (x,y) in enumerate(loader):
      x = to_jax_array(x)
      y= to_jax_array(y)
      x = jnp.transpose(x,(0,2,3,1))
      x_arrays = [
          jax.device_put(x[index], d)
      for d, index in sharding.addressable_devices_indices_map(global_shape_x).items()]
      y_arrays = [
        jax.device_put(y[index], d)
      for d, index in sharding.addressable_devices_indices_map(global_shape_y).items()]
  
      x = jax.make_array_from_single_device_arrays(global_shape_x, sharding, x_arrays)
      y = jax.make_array_from_single_device_arrays(global_shape_y, sharding, y_arrays)


      with mesh:
        state = train_step(state,x,y)
    print(f"Time spent for epoch: {time.time()-t0}")
  
    ggn_vp(loader,state,mesh_sharding,sharding, global_shape_x)


if __name__ == "__main__":
  run(1)

  

  