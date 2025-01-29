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
from jax import vmap
from torch.utils import data
from jax import lax
import os
print(f"Working directory: {os.getcwd()}")



def fft2(x):
   """Applies a 2D FFT over the first two dimensions of the input array x."""
   return fftn(x, axes=(0, 1))


def ifft2(x):
   """Applies a 2D inverse FFT over the first two dimensions of the input array x."""
   return ifftn(x, axes=(0, 1))


class DataGenerator(data.Dataset):
    def __init__(self, u,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u  # Input samples
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key
        self.current_idx = 0  # Track the current index for iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        u = self.__data_generation(subkey)
        return u

    def __len__(self):
        'Return the number of batches'
        return self.N // self.batch_size  # Total full batches

    def __iter__(self):
        'Return an iterator that resets itself'
        self.current_idx = 0  # Reset the index
        return self

    def __next__(self):
        'Get the next batch'
        if self.current_idx >= len(self):  # Stop when all batches are processed
            raise StopIteration
        self.current_idx += 1
        return self.__getitem__(self.current_idx)

    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        u = self.u[idx, :]
        return u





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


    
        
     

    # @jax.jit
    def normalize(self,data):
        min_val = jnp.min(data, axis=(0, 1))
        max_val = jnp.max(data, axis=(0, 1))
        range_val = max_val - min_val
        range_val = jnp.where(range_val == 0, 1, range_val)  # Avoid division by zero
        normalized_data = 2 * (data - min_val) / range_val - 1
        return normalized_data

    
    @partial(jit, static_argnums=(0,))
    def operator_net(self, params, uk):
        if self.arch == 'FNO':
            print(f'uk ko shape before reshape inside operator net.{uk.shape}')
        #    uk = uk.reshape(uk.shape[0], self.N, self.N, uk.shape[3])  # Reshape input 
            input_FNO = uk.reshape(-1, self.N, self.N, 3)  # Reshape for FNO
            print(f'input_fno ko shape after reshape inside operator net.{input_FNO.shape}')


            O = self.N_apply(params, input_FNO)  # Apply the FNO network
            print(f'o ko shape after apply.{O.shape}')

        #    O = jnp.real(O)  # Take the real part of the output
            O = O.reshape(self.N, self.N, 3)  # Reshape output
            print(f'o ko shape after reshape.{O.shape}')


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
      

    def allen_cahn_equation(self, uk, total_steps=500):
            # Expand pp2 and qq2 to include a channel dimension
        self.pp2 = jnp.expand_dims(self.pp2, axis=(0, -1))  # (128, 128) -> (1, 128, 128, 1)
        self.qq2 = jnp.expand_dims(self.qq2, axis=(0, -1))  # (128, 128) -> (1, 128, 128, 1)
        print(f'pp2 shape after expanding: {self.pp2.shape}')  # Expected: (1, 128, 128, 1)

        # Broadcast pp2 and qq2 to match the shape of uk
        self.pp2 = jnp.broadcast_to(self.pp2, (1, self.N, self.N, 3))  # (1, 128, 128, 1) -> (1, 128, 128, 3)
        self.qq2 = jnp.broadcast_to(self.qq2, (1, self.N, self.N, 3))  # (1, 128, 128, 1) -> (1, 128, 128, 3)
        print(f'pp2 shape after broadcasting: {self.pp2.shape}')  # Expected: (1, 128, 128, 3)

        # for _ in range(total_steps):

        cahn = eps**2
        uk = jnp.real(uk)
        print(f'uk ko shape as input:{uk.shape}')

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
        uk = uk.reshape(self.N, self.N, 3)

        

        return jnp.real(uk) # Return only the real part



    @partial(jit, static_argnums=(0,))
    def loss_single(self, params, uk):
        print(f'[uk ko shape on loss single : {uk.shape}')

        u_nn = self.operator_net(params, uk) # predicted or next value of the initial condition
        u_nn = u_nn.reshape(self.N, self.N, 3)
        # print(f'[u_nn ko shape on loss single : {u_nn.shape}')

        u_ac = self.allen_cahn_equation(uk)
        
        # Allen-Cahn equation loss
        # energy_loss = self.total_energy(u_pred)
        datadriven_loss = jnp.mean((u_ac - u_nn) ** 2)
        # total_loss = datadriven_loss + energy_loss
        return datadriven_loss

    @partial(jit, static_argnums=(0,))
    def loss_batches(self, params, batch):
        print(f' batch ko size : {batch.shape}')
        # batch losses
        batch_loss = vmap(self.loss_single, (None, 0))(params, batch)
        batch_loss  = jnp.mean(batch_loss)
        
        return batch_loss


    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, uk):
        params = self.get_params(opt_state)
        print(f'uk ko shape before step:{uk.shape}')
        grads = grad(self.loss_batches)(params, uk)
        return self.opt_update(i, grads, opt_state)


   # Update the train method of the SPiFOL class
    def train(self, data_train, num_epochs=1):
        print(f'data_train ko shape:{data_train[0].shape}')
        # Example training loop
        pbar = trange(num_epochs)
        for it in pbar:
            for batch_idx, batch in enumerate(data_train):
                self.opt_state = self.step(batch_idx, self.opt_state, batch)
                params = self.get_params(self.opt_state)
                loss = self.loss_batches(params, batch)
                self.total_loss_log.append(loss)
                print(f'Epoch: {it}, Batch: {batch_idx}, Loss: {loss}')


        # """Training loop that iterates over all samples in batches."""
        # data_loader = iter(data_generator) # Create an iterable from the data generator
        # print(f'next itercount :{next(self.itercount)}')
        # print(f'data loader:{data_loader}')
        # pbar = trange(nIter)
        # for it in pbar:
        #     try:
        #         batch = next(data_loader)  # Fetch the next batch
        #     except StopIteration:
        #         # Reset the generator if exhausted
        #         data_loader = iter(data_generator)
        #         batch = next(data_loader)

        #     batch = jnp.array(batch)  # Ensure JAX-compatible array
        #     self.opt_state = self.step(next(self.itercount), self.opt_state, batch)

        #     # Logging loss
        #     params = self.get_params(self.opt_state)
        #     loss = self.loss_batches(params, batch)
        #     self.total_loss_log.append(loss)
        #     pbar.set_postfix({'Loss': loss})


            # # Logger (log the loss every 100 iterations)
            # if it % 10 == 0:
            #     params = self.get_params(self.opt_state)
            #     loss = self.loss_batches(params, batch)
            #     self.total_loss_log.append(loss)
            #     # self.total_energy_loss_log.append(energy_loss)
            #     pbar.set_postfix({'Loss': loss})

    #    def train(self, uk, batch_size=32, nIter=1000):
    #         num_batches = uk.shape[0] // batch_size
    #         pbar = trange(nIter)
    #         for it in pbar:
    #             for batch_idx in range(num_batches):
    #                 batch_start = batch_idx * batch_size
    #                 batch_end = batch_start + batch_size
    #                 batch_data = uk[batch_start:batch_end]
    #                 self.opt_state = self.step(next(self.itercount), self.opt_state, batch_data)
    #             if it % 10 == 0:
    #                 params = self.get_params(self.opt_state)
    #                 loss = self.loss_single(params, uk)
    #                 self.total_loss_log.append(loss)
    #                 pbar.set_postfix({'Loss': loss})



    #         # Modify the train method to support batching
    #    def train(self, uk, batch_size=64, nIter=2000):
    #             # Split the data into batches
    #         num_batches = uk.shape[0] // batch_size
    #         pbar = trange(nIter)
        
    #         for it in pbar:
    #             for batch_idx in range(num_batches):
    #                 # Create a batch of data
    #                 batch_start = batch_idx * batch_size
    #                 batch_end = batch_start + batch_size
    #                 batch_data = uk[batch_start:batch_end]  # Slice the batch
                
    #                 # Perform optimization step for the batch
    #                 self.opt_state = self.step(next(self.itercount), self.opt_state, batch_data)

    #             # Logger (log the loss every 10 iterations)
    #             if it % 100 == 0:
    #                 params = self.get_params(self.opt_state)
    #                 loss = self.loss_single(params, uk)  # Compute loss for full dataset
    #                 self.total_loss_log.append(loss)
    #                 pbar.set_postfix({'Loss': loss})






# Parameters
N = 128 # no. of grid points
eps = 0.05 # epsillon 
lr = 0.001 # learning rate
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
input_condition = np.load('./phasefield/u_test.npy')
# input_condition = input_condition[:2000]
# batch_size = 32  # Choose an appropriate batch size
# num_batches = input_condition.shape[0] // batch_size



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
   Permute("ijkl->iljk"),
   FNOBlock2D(32),
   Gelu,  # activation can be changed here
   FNOBlock2D(32),
   Gelu,
   FNOBlock2D(32),
   Permute("ijkl->iklj"),
   Dense(128),
   Gelu,
   Dense(3),
]



cahn = eps**2
# Generate the data trainig samples
data_train = DataGenerator(input_condition, batch_size=20)
print(f'data_train shape: {data_train[0].shape}, Total Batches: {len(data_train)}')

# for i, x in enumerate(data_train):
    # print(f'Batch {i}, Shape: {x.shape}')
# print(len(data_train))  # Should return 100 for 2000 samples and batch size 20.





for arch in arch_list:
#    # Initialize and train the model
    model = SPiFOL(L, x, y, h, eps, pp2, qq2, dt,  N, fno_layers,mlp_layers,lr, arch)
#     # final_state = model.allen_cahn_equation(input_condition)
#     # In your main code:
   
    model.train(data_train, num_epochs=1)
    # batch_size = 64  # Example batch size
    # model.train(input_condition, batch_size=batch_size, nIter=10)





params = model.get_params(model.opt_state)
model_prediction = model.operator_net(params, data_train[0][0])


print("Input Condition Statistics:")
print(f"Min: {jnp.min(input_condition)}, Max: {jnp.max(input_condition)}")
print("Model Prediction Statistics:")
print(f"Min: {jnp.min(model_prediction)}, Max: {jnp.max(model_prediction)}")









# Plot the initial and final states
fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# Input condition plot
axs[0].contour(x, y, jnp.real(input_condition[0].T), levels=[0], colors="black")
axs[0].contour(x, y, jnp.real(model_prediction[0].T), levels=[0], colors="red")

input_condition = input_condition[0, :, :, 0]  # This extracts the 2D array of shape (128, 128)
model_prediction = model_prediction[0, :, :, 0]  # This extracts the 2D array of shape (128, 128)
print(input_condition.shape)  # Expected: (128, 128) or similar
print(model_prediction.shape)  # Expected: (128, 128) or similar
print("Input mean:", jnp.mean(input_condition))
print("Prediction mean:", jnp.mean(model_prediction))





plt_train = jnp.real(data_train[0][0])
plt_train = plt_train[:,:,0]
plt_pred = jnp.real(model_prediction)
plt_pred = plt_pred[:,:,0]

# axs[0].imshow(input_condition)
axs[0].contour(x, y,plt_train , levels=[0], colors="red")
# axs[0].contour(x, y, jnp.real(model_prediction.T), levels=[0], colors="blue")

axs[0].set_title('Input Condition')
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')


axs[1].contour(x, y, plt_pred, levels=[0], colors="blue")
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

plt.plot(model.total_loss_log)
plt.title("Training Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()







plt.tight_layout() 
plt.show()

            