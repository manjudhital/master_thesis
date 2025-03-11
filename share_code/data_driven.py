
import jax
import numpy as np
import jax.random as random
import jax.numpy as jnp
import jax.numpy.fft as jfft
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from numpy import sqrt
from numpy import round
from matplotlib import pyplot as plt
from matplotlib import contour
from jax.numpy.fft import fft2, ifft2
from jax.numpy.fft import fftn, ifftn
from numpy import real
from jax.example_libraries.stax import serial, Gelu
from jax.example_libraries.optimizers import optimizer, make_schedule
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

import os 
# @partial(jit, static_argnums=(0,))
def allen_cahn_equation(uk, pp2, qq2, dt, eps, Nt):

    cahn = eps**2
    samples_timesteps = []

    for iter in range(0, Nt+1):
        uk = jnp.real(uk)

        # Compute denominator in Fourier space
        denominator = cahn + dt * (2 + cahn * (pp2 + qq2))

        # Perform FFT calculations
        s_hat = jfft.fft2(cahn * uk - dt * (uk**3 - 3 * uk)) 

        v_hat = s_hat / denominator  # Now shapes should match


        uk = jfft.ifft2(v_hat)  # inverse FFT
        uk = jnp.real(uk)
        if iter % 1000 == 0:
            samples_timesteps.append(uk)

            # this is to see the how many samples are completed
            print(f'sample {iter} completed')

        # Return the real part
    return jnp.array(samples_timesteps)  # Return only the real part





# difinning the no of grid points in x, y and z
Nx = 28 # number of grid points in x be positive even integer number
Ny = 28 # number of grid points in y be positive even integer number



# Define the parameters of the Allen-Cahn equation in 2d
Lx = 2.0 * jnp.pi #length of the domain in x
Ly = 2.0 * jnp.pi #length of the domain in y
hx = Lx / Nx #spatial step size in coordinate x
hy = Ly / Ny #spatial step size in coordinate y
dt = 0.0001 #time step size
T = 4 #final time
Nt = int(jnp.round(T/dt)) #number of time steps
ns = Nt / 10 #number of snapshots

# Define the grid points in x and y direction
def x_gridpoint(Nx, Lx, hx):
    x = jnp.linspace(-0.5*Lx+hx,0.5*Lx,Nx)
    return x
x = x_gridpoint(Nx, Lx, hx) #number of grid points in x direction and step size and limitation on x  axis
def y_gridpoint(Ny, Ly, hy):
    y = jnp.linspace(-0.5*Ly+hy,0.5*Ly,Ny)
    return y
y = y_gridpoint(Ny, Ly, hy) #number of grid points in y direction and step size and limitation on y  axis 

# creating meshgrid in x and y direction
xx,yy = jnp.meshgrid(x,y) #creating meshgrid in x and y direction 

epsillon = 0.5 #small parameter # interface thickness in the Allen-Cahn equation 
cahn = epsillon**2 #cahn number  

# theta = jnp.arctan2(yy, xx)
#   # or another appropriate value
# uk = jnp.tanh((1.7 + 1.2 * np.cos(6 * theta)) - jnp.sqrt(xx**2 + yy**2) / (jnp.sqrt(2) * epsillon))
data = np.load('data_generation_checking/phasefield2d_data_28x28_10k.npy')

# Select 1,000 random samples
key = jax.random.PRNGKey(0)  # Random seed for reproducibility
idx = jax.random.choice(key, data.shape[0], shape=(1000,), replace=False)  # Random 1k indices
input_samples = data[idx]  # Shape: (1000, Nx, Ny)
# print(f'uk ko shape:{uk.shape}')


# defining the wavenumber in x and y direction , which is in fourier space
p = jnp.concatenate([2 * jnp.pi / Lx * jnp.arange(0, Nx//2), 2 * jnp.pi / Lx * jnp.arange(-Nx//2  , 0)]) # wavenumber in x direction
q = jnp.concatenate([2 * jnp.pi / Ly * jnp.arange(0, Ny//2), 2 * jnp.pi / Ly * jnp.arange(-Ny//2 , 0)])


# square of wavenumber in x and y direction
p2 = p**2 # square of wavenumber in x direction
q2 = q**2 # square of wavenumber in y direction

# creating meshgrid in x and y direction for square of wavenumber
pp2, qq2 = jnp.meshgrid(p2, q2)


input_samples= input_samples.reshape(-1, Nx , Ny)

samples = []

# this for to see the how many samples are printed 
for i, uk in enumerate (input_samples):
    print(f'sample {i} started')

# for uk in input_samples:
   
    ac_input = allen_cahn_equation(uk, pp2, qq2, dt, epsillon, Nt)
# print(f'shape of ac_input:{ac_input.shape}')
    samples.append(ac_input)
samples = jnp.array(samples)
# print(f'samples ko shape:{samples.shape}')
      

# Specify the directory where y want to save the data
save_dir = './data_driven/data'

    # Ensure the directory exists, create it if not
os.makedirs(save_dir, exist_ok=True)

#saving the data 
# Save the training and testing data
# print("Saving training data to u_train.npy...")
np.save(os.path.join(save_dir, "driven_data_28x28_1k_input_samples_20timestep_every2kiter.npy"), np.array(input_samples))
np.save(os.path.join(save_dir, "driven_data_28x28_1ksample_20timestep_every2kiter.npy"), np.array(samples))

# # loaded the generated data
# loaded_input_samples = np.load(os.path.join(save_dir, "driven_data_28x28_1k_input_samples_20timestep_every2kiter.npy"))
# print(f'loaded_input_samples ko shape:{loaded_input_samples.shape}')

# loaded_samples = np.load(os.path.join(save_dir, "driven_data_28x28_1ksample_20timestep_every2kiter.npy"))
# print(f'loaded_samples ko shape:{loaded_samples.shape}')