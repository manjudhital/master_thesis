import jax
import jax.numpy as jnp
from jax import random
from jax.example_libraries.stax import serial, Gelu
from jax.example_libraries.optimizers import optimizer, make_schedule

# Complex Adam optimizer
@optimizer
def complex_adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
    """Construct optimizer triple for complex-valued Adam."""
    step_size = make_schedule(step_size)

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        g = jnp.conj(g)  # Complex conjugate
        m = (1 - b1) * g + b1 * m  # First moment
        v = (1 - b2) * jnp.real(jnp.conj(g) * g) + b2 * v  # Second moment
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v

    def get_params(state):
        x, m, v = state
        return x

    return init, update, get_params


# Define Dense Layer
def Dense(out_dim, W_init=jax.nn.initializers.glorot_uniform(), b_init=jax.nn.initializers.normal()):
    """Layer constructor function for a dense (fully-connected) layer."""
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return jnp.dot(inputs, W) + b

    return init_fun, apply_fun


# Define FNO Block for Spatio-Temporal Data
def FNOBlock3D(modes):
    def compl_mul3d(input, weights):
        return jnp.einsum("jilxyz,iklxyz->jklxyz", input, weights)
    
    # def compl_mul3d(input, weights):
    #     return jnp.einsum("bctxyz,coxyz->btoxyz", input, weights)

    def init_fun(rng, input_shape):
        k1, k2 = random.split(rng)
        W1 = random.normal(k1, (input_shape[1], input_shape[1], modes, modes, modes))
        W2 = random.normal(k2, (input_shape[1], input_shape[1], modes, modes, modes))
        return input_shape, (W1, W2)

    def apply_fun(params, inputs, **kwargs):
        W1, W2 = params
        x_ft = jnp.fft.rfftn(inputs, axes=(-3, -2, -1))  # Apply FFT in time, height, and width
        out_ft = jnp.zeros_like(x_ft)
        
        # Apply weights to lower frequency components
        out_ft = out_ft.at[:, :, : W1.shape[2], : W1.shape[3], : W1.shape[4]].set(
            compl_mul3d(x_ft[:, :, : W1.shape[2], : W1.shape[3], : W1.shape[4]], W1)
        )
        
        # Apply weights to higher frequency components
        out_ft = out_ft.at[:, :, -W2.shape[2] :, : W2.shape[3], : W2.shape[4]].set(
            compl_mul3d(x_ft[:, :, -W2.shape[2] :, : W2.shape[3], : W2.shape[4]], W2)
        )
        
        return jnp.fft.irfftn(out_ft, s=inputs.shape[-3:])  # Inverse FFT to get back to time-space domain
    
    return init_fun, apply_fun


# Define Permute Layer
def Permute(order):
    def permutation_indices(order):
        if order == "ijklm->imjkl":
            return (0, 4, 1, 2, 3)
        elif order == "ijklm->iklmj":
            return (0, 2, 3, 4, 1)
        else:
            raise NotImplementedError

    def init_fun(rng, input_shape):
        idx = permutation_indices(order)
        output_shape = tuple([input_shape[i] for i in idx])
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        outputs = jnp.einsum(order, inputs)
        return outputs

    return init_fun, apply_fun



# Define MLP architecture
def MLP(layers, activation=jax.nn.relu):
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 6.0)
            W = glorot_stddev * random.normal(key, (d_in, d_out))
            b = jnp.zeros(d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = [init_layer(k, layers[i], layers[i + 1]) for i, k in enumerate(keys)]
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = jnp.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        return jnp.dot(inputs, W) + b

    return init, apply


# Define modified MLP
def modified_MLP(layers, activation=jax.nn.relu):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.0)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    def init(rng_key):
        U1, b1 = xavier_init(random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 = xavier_init(random.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = [init_layer(k, layers[i], layers[i + 1]) for i, k in enumerate(keys)]
        return (params, U1, b1, U2, b2)

    def apply(params, inputs):
        params, U1, b1, U2, b2 = params
        U = activation(jnp.dot(inputs, U1) + b1)
        V = activation(jnp.dot(inputs, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(jnp.dot(inputs, W))
            inputs = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V)
        W, b = params[-1]
        return jnp.dot(inputs, W) + b

    return init, apply



