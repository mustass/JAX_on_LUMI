import torch
import torchvision
import torchvision.transforms as transforms
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from jax import pmap

# Define a simple convolutional neural network
class CNN(nn.Module):
    num_classes: int

    def setup(self):
        self.conv1 = nn.Conv(32, (3, 3), padding='SAME')
        self.conv2 = nn.Conv(64, (3, 3), padding='SAME')
        self.fc1 = nn.Dense(128)
        self.fc2 = nn.Dense(self.num_classes)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.max_pool(x, (2, 2))
        x = nn.relu(self.conv2(x))
        x = nn.max_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.relu(self.fc1(x))
        return self.fc2(x)

# JAX functions to convert numpy arrays to jax arrays
def to_jax_array(np_array):
    return jnp.array(np_array)

# Set up the CIFAR-100 dataset with PyTorch DataLoader
def get_dataloader(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader

# Define a loss function
def loss_fn(params, model, x, y):
    logits = model.apply({'params': params}, x)
    return jax.nn.softmax_cross_entropy(logits=logits, labels=jax.nn.one_hot(y, 100))

# Training function
@pmap
def train_step(params, x, y):
    grads = jax.grad(loss_fn)(params, model, x, y)
    return grads

# Main training loop
def train_model(num_epochs=5, batch_size=64):
    # Get the DataLoader
    dataloader = get_dataloader(batch_size)
    
    # Initialize model and optimizer
    model = CNN(num_classes=100)
    rng = jax.random.PRNGKey(0)
    input_shape = (batch_size, 3, 32, 32)  # CIFAR-100 images
    params = model.init(rng, jnp.ones(input_shape))['params']
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)

    # Training loop
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # Convert inputs and labels to JAX arrays
            inputs = to_jax_array(inputs.numpy())
            labels = to_jax_array(labels.numpy())
            
            # Split data across devices
            inputs = jnp.reshape(inputs, (jax.local_device_count(), -1, 3, 32, 32))
            labels = jnp.reshape(labels, (jax.local_device_count(), -1))

            # Perform a training step
            grads = train_step(params, inputs, labels)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if i % 100 == 0:
                loss = loss_fn(params, model, inputs[0], labels[0])  # Use one device for logging
                print(f'Epoch {epoch}, Step {i}, Loss: {loss}')

# Start training
if __name__ == '__main__':
    train_model(num_epochs=5, batch_size=64)
