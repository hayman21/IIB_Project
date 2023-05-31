import numpy as np
import matplotlib.pyplot as plt
from Base_Files.ClassLevyJumpProcesses import TemperedStableSubordinator

# Set Parameters
t1 = 0.0
t2 = 10.0
num_obs = 100
num_epochs = 2000
subordinator_truncation = 0.0
kappa = 0.7
delta = 1.5
gamma = 1.0
nProcesses = 1

# Generate data
g_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma)
initial_g= g_sub.generate_path()
#plt.plot(np.linspace(t1, t2, num_obs), initial_g)
#plt.title("Aim Distribution")

# Define a log_likelihood function
#def log_likelihood(data, mu=0, sigma=1):
    #return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (data - mu) ** 2 / sigma ** 2)
#def log_likelihood(data):
   # return np.log(multivariate_normal(data, mean=0, cov=GaussianKernel(theta, l=1)))
# Compute initial likelihood
#def log_likelihood(data, N, l):
   # D = squareform(pdist(data.reshape(-1,1)))
   # K = np.exp(-D ** 2 / (2 * l ** 2))
   # z = np.random.randn(N)
    #L = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    #u = L @ z
   # log_likelihood = -0.5 * z.T @ z - np.sum(np.log(np.diag(L)))
   # return log_likelihood



# Define the number of iterations and burn-in period
num_iter = 100
burn_in = 0

# Run the Metropolis-Hastings algorithm
samples = []
for i in range(num_iter):
    # Draw a new proposal from the proposal distribution
    theta = initial_g
    # initial settings t1=0 t2=1.0 num_obs=500
    theta_star = TemperedStableSubordinator(0.0, 10.0, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path()

    # Compute the acceptance probability
    log_alpha = log_likelihood(theta_star, N=num_obs, l=1) - log_likelihood(theta, N=num_obs, l=1)
    alpha = np.exp(log_alpha)
    print(log_likelihood(theta, N=num_obs, l=1))
    print(log_likelihood(theta_star, N=num_obs, l=1))
    print(log_alpha)
    print(alpha)
    accept = np.random.uniform() < alpha

    # Update the current value of the subordinator
    if accept:
        theta = theta_star

    # Add the current value to the list of samples
    samples.append(theta)


# Discard the burn-in period
samples = samples[burn_in:]
print(samples)
X = np.linspace(t1, t2, num_obs).reshape(-1, 1)
plt.plot(X, initial_g, label='initial')
plt.plot(X, samples[-1], label='test')
plt.legend()
plt.show()
#for i in range(3):
#    plt.plot(np.linspace(0, 1, 500), g_sub.generate_path(), label= "Run_" +str(i+1))

#plt.plot(np.linspace(t1, t2, num_obs), initial_g, label='Initial')
#plt.legend()


#plt.show()


#plt.plot(g_sub.generate_path())
#plt.show()