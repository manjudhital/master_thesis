import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.example_libraries.stax import Dense, Gelu, serial
from jax.example_libraries.optimizers import optimizer, make_schedule
# from jax.scipy.fftpack import fftn, ifftn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from functools import partial
from jax.numpy.fft import fftn, ifftn, fftshift, ifftshift
from jax.example_libraries.optimizers import exponential_decay
import jax.numpy.fft as jfft
from jax.example_libraries.stax import Dense, Gelu, serial, glorot_normal
from spifol_archs import FNOBlock2D, Permute, complex_adam, MLP, modified_MLP




def fft2(x):
   """Applies a 2D FFT over the first two dimensions of the input array x."""
   return fftn(x, axes=(0, 1))


def ifft2(x):
   """Applies a 2D inverse FFT over the first two dimensions of the input array x."""
   return ifftn(x, axes=(0, 1))




# Define normalization and denormalization functions
def normalize(data):
   mean = jnp.mean(data)
   std = jnp.std(data)
   return (data - mean) / std, mean, std


def denormalize(data, mean, std):
   return data * std + mean



class SPiFOL:
   def __init__(self, L, x, y, h, eps, pp2, qq2, dt,  N, fno_layers, mlp_layers,lr, arch):
       self.arch = arch
       self.N = N
       self.lr = lr
       # self.norm_par = norm_par
       self.eps = eps
       self.pp2 = pp2
       self.qq2 = qq2
       self.dt = dt
       self.L = L
       self.h = h
       self.x = x
       self.y = y
       # Initialize the network based on architecture type
       if arch == 'FNO':
           self.N_init, self.N_apply = serial(*fno_layers)
           _, params = self.N_init(random.PRNGKey(1234), (-1, N, N, 3))
          
       elif arch == 'MLP':
           self.N_init, self.N_apply = MLP(mlp_layers)
           params = self.N_init(random.PRNGKey(1234))
          
       elif arch == 'modified_MLP':
           self.N_init, self.N_apply = modified_MLP(mlp_layers)
           params = self.N_init(random.PRNGKey(1234))
       else:
           raise ValueError("Unsupported architecture!")


       self.params = params


    
       # Optimizer setup
       self.opt_init, self.opt_update, self.get_params = complex_adam(
           jax.example_libraries.optimizers.exponential_decay(
               lr, decay_steps=2000, decay_rate=0.9)
           )
      
       self.opt_state = self.opt_init(self.params)


       # Logging losses
       self.total_loss_log = []
       self.total_energy_loss_log = []


       # Initialize optimizer state
       self.opt_state = self.opt_init(self.params)
       self.itercount = iter(range(50000))


   @partial(jit, static_argnums=(0,))
   def operator_net(self, params, uk):
       if self.arch == 'FNO':
        #    print(f'uk ko shape before reshape inside operator net.{uk.shape}')
           uk = uk.reshape(uk.shape[0], self.N, self.N, uk.shape[3])  # Reshape input 
           O = self.N_apply(params, uk)  # Apply the FNO network
        #    print(f'o ko shape after network output.{O.shape}')

           O = jnp.real(O)  # Take the real part of the output
           O = O.reshape(uk.shape[0],self.N, self.N, uk.shape[3])  # Reshape output
        #    print(f'o ko shape after reshape.{O.shape}')

           return O
       elif self.arch == 'MLP':
           uk = uk.flatten()
           O = self.N_apply(params, uk)  # Directly apply the network
           O = O.reshape(uk.shape[0], self.N, self.N, uk.shape[3])  # Reshape output to match strain components
           # O = O / self.norm_par  # Normalize the output
           return O
       elif self.arch == 'modified_MLP':
           uk = uk.flatten()
           O = self.N_apply(params, uk)
           O = O.reshape(uk.shape[0], self.N, self.N, uk.shape[3])
           return O
       else:
           raise ValueError("Unsupported architecture type!")
      

   def allen_cahn_equation(self, uk, total_steps=700):
         # Expand pp2 and qq2 to include a channel dimension
        self.pp2 = jnp.expand_dims(self.pp2, axis=(0, -1))  # (128, 128) -> (1, 128, 128, 1)
        self.qq2 = jnp.expand_dims(self.qq2, axis=(0, -1))  # (128, 128) -> (1, 128, 128, 1)

        # Broadcast pp2 and qq2 to match the shape of uk
        self.pp2 = jnp.broadcast_to(self.pp2, (uk.shape[0], self.N, self.N, uk.shape[3]))  # (1, 128, 128, 1) -> (1, 128, 128, 3)
        self.qq2 = jnp.broadcast_to(self.qq2, (uk.shape[0], self.N, self.N, uk.shape[3]))  # (1, 128, 128, 1) -> (1, 128, 128, 3)

        for _ in range(total_steps):

            cahn = eps**2
            uk = jnp.real(uk)

            # Check shapes for debugging
            # print(f"pp2 shape after broadcasting: {self.pp2.shape}")  # Expected: (1, 128, 128, 3)
            # print(f"qq2 shape after broadcasting: {self.qq2.shape}")  # Expected: (1, 128, 128, 3)
            # print(f"uk shape before broadcasting: {uk.shape}")  # Expected: (1, 128, 128, 3)

            # Compute denominator in Fourier space
            denominator = cahn + self.dt * (2 + cahn * (self.pp2 + self.qq2))  # Shape: (1, 128, 128, 3)
            # print(f"denominator shape: {denominator.shape}")  # Expected: (1, 128, 128, 3)

            # Perform FFT calculations
            s_hat = jfft.fft2(cahn * uk - self.dt * (uk**3 - 3 * uk))  # Shape: (1, 128, 128, 3)
            v_hat = s_hat / denominator  # Shape: (1, 128, 128, 3)
            uk = jfft.ifft2(v_hat)  # Shape: (1, 128, 128, 3)

            # Check results
            # print(f"s_hat shape: {s_hat.shape}")  # Expected: (1, 128, 128, 3)
            # print(f"v_hat shape: {v_hat.shape}")  # Expected: (1, 128, 128, 3)
            # print(f"uk shape after computation: {uk.shape}")  # Expected: (1, 128, 128, 3)

        return jnp.real(uk)  # Return only the real part
   


   @partial(jit, static_argnums=(0,))
   def loss_single(self, params, uk):
        u_nn = self.operator_net(params, uk) # predicted or next value of the initial condition
        u_nn = u_nn.reshape(-1, self.N, self.N, uk.shape[3])
        # print(f'[u_nn ko shape : {u_nn.shape}')

        u_ac = self.allen_cahn_equation(uk)
        
        # Allen-Cahn equation loss
    #    energy_loss = self.total_energy(u_pred)
        datadriven_loss = jnp.mean((u_ac - u_nn) ** 2)
    #    total_loss = distance_loss + energy_loss
        return datadriven_loss
   

   @partial(jit, static_argnums=(0,))
   def step(self, i, opt_state, uk):
        params = self.get_params(opt_state)
        grads = grad(self.loss_single)(params, uk)
        return self.opt_update(i, grads, opt_state)


   def train(self, uk, nIter=1000):
        pbar = trange(nIter)
        for it in pbar:
            self.opt_state = self.step(next(self.itercount), self.opt_state, uk)
    #    self.itercount = iter(range(nIter))  # Create an iterator
            



            # Logger (log the loss every 100 iterations)
            if it % 100 == 0:
                params = self.get_params(self.opt_state)
                loss = self.loss_single(params, uk)
                self.total_loss_log.append(loss)
                # self.total_energy_loss_log.append(energy_loss)
                pbar.set_postfix({'Loss': loss})


# Parameters
N = 128 # no. of grid points
eps = 0.05 # epsillon 
lr = 0.1 # learning rate
dt = 0.0001 # time step or time increment
L = 2 * jnp.pi # length of domian
h = L/N # spacing between grid or length of grid





x = jnp.linspace(-0.5 * L + h, 0.5 * L, N)
y = jnp.linspace(-0.5 * L + h, 0.5 * L, N)


theta = jnp.arctan2(y, x)
# Generate input condition
xx, yy = jnp.meshgrid(x, y)
# input_condition = jnp.tanh((2 - jnp.sqrt(xx**2 + yy**2)) / jnp.sqrt(2)* eps)
# input_condition = np.load (os.path.join(save_dir, "u_train.npy"), np.array(data.u_train))
# Load the .npy file
input_condition = np.load('data_generation/u_test.npy')
input_condition = input_condition[:2000]

# Inspect the data
# print(input_condition)
# print(type(input_condition)) 


 # defining the wavenumber in x and y direction , which is in fourier space
p = jnp.concatenate([2 * jnp.pi / L * jnp.arange(0, N//2), 2 * jnp.pi / L * jnp.arange(-N//2  , 0)]) # wavenumber in x direction
q = jnp.concatenate([2 * jnp.pi / L * jnp.arange(0, N//2), 2 * jnp.pi / L * jnp.arange(-N//2 , 0)])




# # square of wavenumber in x and y direction
p2 = p**2 # square of wavenumber in x direction
q2 = q**2 # square of wavenumber in y direction


# # creating meshgrid in x and y direction for square of wavenumber
pp2, qq2 = jnp.meshgrid(p2, q2)



# arch_list = ['FNO', 'MLP', 'modified_MLP']
arch_list = ['FNO']
# arch = 'modified_MLP'
# mlp layers
mlp_layers = [16384, 32, 32, 16384]


# Define FNO layers
fno_layers = [
   Dense(32),
#    Permute("ijkl->iljk"),
#    FNOBlock2D(32),
   Gelu,  # activation can be changed here
#    FNOBlock2D(32),
#    Gelu,
#    FNOBlock2D(32),
#    Permute("ijkl->iklj"),
   Dense(32),
   Gelu,
   Dense(3),
]



cahn = eps**2
for arch in arch_list:
   # Initialize and train the model
    model = SPiFOL(L, x, y, h, eps, pp2, qq2, dt,  N, fno_layers,mlp_layers,lr, arch)
    # final_state = model.allen_cahn_equation(input_condition)
    model.train(input_condition, nIter=5000)


params = model.get_params(model.opt_state)
model_prediction = model.operator_net(params, input_condition)



# Plot the initial and final states
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# Input condition plot
# axs[0].contour(x, y, jnp.real(input_condition[0].T), levels=[0], colors="black")
# axs[0].contour(x, y, jnp.real(model_prediction[0].T), levels=[0], colors="red")

input_condition = input_condition[0, :, :, 0]  # This extracts the 2D array of shape (128, 128)
model_prediction = model_prediction[0, :, :, 0]  # This extracts the 2D array of shape (128, 128)



# axs[0].imshow(input_condition)
axs[0].contour(x, y, jnp.real(input_condition.T), levels=[0], colors="red")
# axs[0].contour(x, y, jnp.real(model_prediction.T), levels=[0], colors="blue")

axs[0].set_title('Input Condition')
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')


axs[1].contour(x, y, jnp.real(model_prediction.T), levels=[0], colors="blue")
# axs[1].imshow(model_prediction)
axs[1].set_title('Evolved condition ')
axs[1].set_aspect('equal', adjustable='box')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')


# #    # Plot loss
axs[2].plot(model.total_loss_log)
axs[2].set_title('Training Loss')
axs[2].set_xlabel('Iteration')
axs[2].set_ylabel('Loss')






plt.tight_layout()
plt.show()

            