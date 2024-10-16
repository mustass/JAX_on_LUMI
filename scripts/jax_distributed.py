# The following is run in parallel on each host on a GPU cluster or TPU pod slice.
import jax
jax.distributed.initialize()  # On GPU, see above for the necessary arguments.  # total number of accelerator devices in the cluste
print(f"Jax devices: {jax.device_count()} are [{jax.devices()}]")
print(jax.local_device_count())  # number of accelerator devices attached to this ho
# The psum is performed over all mapped devices across the pod slice
xs = jax.numpy.ones(jax.local_device_count())
print(jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs))