import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from Base_Files.ClassLevyJumpProcesses import TemperedStableSubordinator
from Base_Files.Creating_the_NGP import GaussianProcess
import GPy

# Set Parameters
t1 = 0.0
t2 = 10.0
num_obs = 500 # (N) number of points e.g. size of data set
num_epochs = 2000
subordinator_truncation = 0.0
kappa = 0.7
delta = 1.5
gamma = 1.0
nProcesses = 1
l = 1

# Subordinator
initial_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path()
NGP_data = GaussianProcess(initial_sub.reshape(-1,1), l, num_obs)
#print(initial_sub.reshape(-1,1))

# Creating the Non-Gaussian Process
D = squareform(pdist(initial_sub.reshape(-1,1)))                        # Calculating the distance between points
K = np.exp(-D**2/(2*l**2))                                              # Covariance matrix
z = np.random.randn(num_obs)                                            # Generate a vector of Gaussian vales - N(0, I)
L = np.linalg.cholesky(K + 1e-6 * np.eye(num_obs))                      # Calculate the Cholesky decomposition of K. This is the matric such that K = L L^T
u = L @ z                                                               # Matrix multiply L and z

#fig, ax = plt.subplots(nrows=2, figsize=(12,8))
#ax[0].plot(np.linspace(t1, t2, num_obs), u)
#ax[0].set_xlabel('x', fontsize=15)
#ax[0].set_ylabel('y(x)', fontsize=15)
#ax[0].grid(True)

#ax[1].plot(np.linspace(t1, t2, num_obs), initial_sub, label='Initial Subordinator')
#ax[1].set_xlabel('x', fontsize=15)
#ax[1].set_ylabel('W(x)', fontsize=15)
#ax[1].grid(True)
#plt.legend()
#plt.tight_layout()
#plt.show()

def log_likelihood(X, Y):
    # Define your Gaussian process model with a subordinator
    kernel = GPy.kern.RBF(input_dim=1)
    sub_kernel = GPy.kern.Brownian(input_dim=1)
    model = GPy.models.GPRegression(X, Y, kernel * sub_kernel)

    # Calculate the log-likelihood of the model
    log_likelihood = model.log_likelihood()
    return log_likelihood
# Define a log_likelihood function
#def log_likelihood_gp(subord, y):
 #   n = num_obs
  #  D = squareform(pdist(subord.reshape(-1, 1)))
   # K = np.exp(-D**2/(2*l**2))
    #L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    #alpha = cho_solve((L, True), y)
    #log_likelihood = -0.5 * np.dot(y, alpha) - np.sum(np.log(np.diag(L))) \
    #                 - 0.5 * n * np.log(2 * np.pi)
    #return log_likelihood
#print("Log likelihood:",log_likelihood_gp(initial_sub, u))

# Define the number of iterations and burn-in period
num_iter = 500
burn_in = 0

# Run the Metropolis-Hastings algorithm
samples = []
estimates = []
likelihood_samples = []
alphas = []
accepts = []
likelihood_accept = []
current_sub = initial_sub
for i in range(num_iter):
    # Draw a new proposal from the proposal distribution
    #theta = u
    print(i)
    # initial settings t1=0 t2=1.0 num_obs=500
    new_sub = TemperedStableSubordinator(0.0, 10.0, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path()
    theta_star = GaussianProcess(new_sub.reshape(-1,1), l, num_obs)

    # Compute the acceptance probability
    log_alpha = log_likelihood(new_sub.reshape(-1,1), theta_star) - log_likelihood(current_sub.reshape(-1,1), NGP_data)
    alpha = np.exp(log_alpha)
    #print(log_likelihood_gp(initial_sub, theta))
    #print(log_likelihood_gp(new_sub, theta_star))
    #print(log_alpha)
    #print(alpha)
    accept = np.random.uniform() < alpha
    alphas.append(alpha)
    # Update the current value of the subordinator
    if accept:
        current_sub = new_sub
        #likelihood_accept.append(log_likelihood(current_sub, GaussianProcess(current_sub.reshape(-1, 1), l, num_obs)))
    samples.append(current_sub)
    #likelihood_samples.append(log_likelihood(current_sub, GaussianProcess(current_sub.reshape(-1,1), l, num_obs)))
        #if i == 0:
         #   samples.append(new_sub)
          #  likelihood_samples.append(log_alpha)
        #else:
         #   if log_alpha > likelihood_samples[-1]:
           #     samples.append(new_sub)
          #      likelihood_samples.append(log_alpha)
   # else:
       # likelihood_samples.append(likelihood_samples[-1])

    #samples.append(initial_sub)
       #likelihood_samples.append(log_alpha)

    # Add the current value to the list of samples



# Discard the burn-in period
#samples = samples[burn_in:]
#print(samples)
print(likelihood_samples)
print(len(likelihood_samples))
X = np.linspace(t1, t2, num_obs).reshape(-1, 1)

fig, ax = plt.subplots(nrows=3, figsize=(12,8))
ax[0].plot(X, u, label='initial')
    #ax[0].plot(X, estimates[-1], label='test')
ax[0].set_xlabel('x', fontsize=15)
ax[0].set_ylabel('y(x)', fontsize=15)
ax[0].grid(True)

    #plt.legend()

ax[1].plot(X, initial_sub, label='Initial')
#for k in range(5):
ax[1].plot(X, samples[-1], label='Estimate')
#ax[1].plot(X, samples[-1], label='Estimate')
ax[1].title.set_text('Subordinator')
ax[1].set_xlabel('x', fontsize=15)
ax[1].set_ylabel('W(x)', fontsize=15)
ax[1].grid(True)

ax[2].plot(np.linspace(1, len(likelihood_samples),len(likelihood_samples)), likelihood_samples, label='likelihood')
ax[2].grid(True)

print(likelihood_samples)

#print(len(likelihood_samples))
#print(likelihood_samples)
#Y = np.linspace(1, len(likelihood_samples),len(likelihood_samples))
#plt.plot(Y, likelihood_samples)
#print(np.linspace(1, num_iter, num_iter))
#plt.show()

#plt.figure(1)
#plt.plot(X, u, label='initial')
#plt.plot(X, GaussianProcess(samples[-1].reshape(-1,1), l, num_obs), label='test')

#plt.figure(2)
#plt.plot(X, initial_sub, label='Initial Subordinator')
#plt.plot(X, samples[-1], label='Test Subordinator')

plt.legend()
plt.tight_layout()
plt.show()

