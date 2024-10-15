import flax
import jax
import jax.numpy as jnp
import optax
import functools
class Trainer:

    def __init__(self, model, optimizer, rng, x_sharding, state_sharding, y_sharding):
        self.model = model
        self.optimizer = optimizer
        self.create_funcs(state_sharding, x_sharding, y_sharding)
        self.rng = rng


    def create_funcs(self, state_sharding, x_sharding, y_sharding):
        
        @functools.partial(jax.jit, in_shardings=(state_sharding, x_sharding,y_sharding),
                   out_shardings=state_sharding)
        
        def train_step(opt_state,x,y):

            def compute_loss(params, x, y):
                logits = self.model.apply({'params': params}, x)
                return jax.nn.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(y, 100))

            def loss_fn(params):
                return compute_loss(params, x, y)

            grads = jax.grad(loss_fn)(opt_state.params)
            opt_state = opt_state.apply_gradients(grads=grads)
            return opt_state
        self.train_step = train_step

    # Train ste

    # Main training loop
    def train_model(self, params, num_epochs=5):
        # Prepare dataset and optimizer
        # Initialize model and optimizer parameters
        opt_state = self.optimizer.init(params)

        # Training loop
        for e in range(num_epochs):
            print(f"Epoch {e}")
            for (inputs,labels) in self.train_loader:

                inputs = jnp.reshape(inputs, (jax.local_device_count(), -1, 32, 32,3))
                labels = jnp.reshape(labels, (jax.local_device_count(), -1))

                #batch = (inputs,labels)
                params, opt_state = self.train_step(opt_state, params,  inputs,labels)
            
        return params

