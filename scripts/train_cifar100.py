import torch
import torchvision
import torchvision.transforms as transforms
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import pmap
from lumihax.training.base_trainer import Trainer
from typing import Callable
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
from flax.training import train_state, checkpoints
import functools

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
  return NamedSharding(mesh, pspec)
# Define a simple convolutional neural network
class CNN(nn.Module):
    num_classes: int
    dense_init: Callable = nn.initializers.xavier_normal()

    def setup(self):
        self.conv1 = nn.Conv(32, (3, 3), padding='SAME', kernel_init=nn.with_partitioning(self.dense_init, ('data', None)))
        self.conv2 = nn.Conv(64, (3, 3), padding='SAME', kernel_init=nn.with_partitioning(self.dense_init, ('data', None)))
        self.fc1 = nn.Dense(128,kernel_init=nn.with_partitioning(self.dense_init, ('data', None)),
                            use_bias=False
                 )
        self.fc2 = nn.Dense(self.num_classes,kernel_init=nn.with_partitioning(self.dense_init, ('data', None)),
                            use_bias=False,
                 )

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = with_sharding_constraint(x,mesh_sharding(PartitionSpec('data',None)))
        x = nn.max_pool(x, (2, 2))
        x = with_sharding_constraint(x,mesh_sharding(PartitionSpec('data',None)))
        x = nn.relu(self.conv2(x))
        x = with_sharding_constraint(x,mesh_sharding(PartitionSpec('data',None)))
        x = nn.max_pool(x, (2, 2))
        x = with_sharding_constraint(x,mesh_sharding(PartitionSpec('data',None)))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.relu(self.fc1(x))
        x = with_sharding_constraint(x,mesh_sharding(PartitionSpec('data',None)))
        x = self.fc2(x)
        x = with_sharding_constraint(x,mesh_sharding(PartitionSpec('data',None)))
        
        return x, None

# Set up the CIFAR-100 dataset with PyTorch DataLoader
def get_dataloader(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root='/scratch/project_465001020/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def init_fn(k, x, model, optimizer):
  variables = model.init(k, x) # Initialize the model.
  state = train_state.TrainState.create( # Create a `TrainState`.
    apply_fn=model.apply,
    params=variables['params'],
    tx=optimizer)
  return state

# Main training loop
def train_model(mesh, num_epochs=5, batch_size=64):
    # Get the DataLoader
    dataloader = get_dataloader(batch_size)
    
    # Initialize model and optimizer
    model = CNN(num_classes=100)
    rng = jax.random.PRNGKey(0)
    optimizer = optax.adam(learning_rate=0.001)
    x = jnp.ones((64,32,32,3))
    x_sharding = mesh_sharding(PartitionSpec('data', None)) # dimensions: (batch, length)
    x = jax.device_put(x, x_sharding)

    abstract_variables = jax.eval_shape(
    functools.partial(init_fn, model=model, optimizer=optimizer), rng, x)
    state_sharding = nn.get_sharding(abstract_variables, mesh)
    jit_init_fn = jax.jit(init_fn, static_argnums=(2, 3),
                      in_shardings=(mesh_sharding(()), x_sharding),  # PRNG key and x
                      out_shardings=state_sharding)

    initialized_state = jit_init_fn(rng, x, model, optimizer)
    print("Starting")
    trainer = Trainer(model, optimizer,dataloader,rng,x_sharding,state_sharding)
    for (x,y) in dataloader:
        trained_params = trainer.train_step(initialized_state, x,y)


        
if __name__ == "__main__":
    device_mesh = mesh_utils.create_device_mesh((2,))
    mesh = Mesh(devices=device_mesh, axis_names=('data',))
    print(mesh)


    train_model(mesh,10,64)
