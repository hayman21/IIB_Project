import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from Base_Files.ClassLevyJumpProcesses import TemperedStableSubordinator
from scipy.linalg import cho_solve
from Base_Files.Creating_the_NGP import GaussianProcess

# Set Parameters
t1 = 0.0
t2 = 100.0
num_obs = 1000 # (N) number of points e.g. size of data set
num_epochs = 2000
subordinator_truncation = 0.0
kappa = 0.7
delta = 1.5
gamma = 1.0
nProcesses = 1
l = 1

initial_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path()

X = initial_sub
Y = GaussianProcess(initial_sub.reshape(-1,1), 1, num_obs)

def log_likelihood(X, Y):
    n = num_obs
    D = squareform(pdist(X.reshape(-1, 1)))
    K = np.exp(-D**2/(2*l**2))
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    alpha = cho_solve((L, True), Y)
    log_likelihood = -0.5 * np.dot(Y, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
    return log_likelihood

num_iter = 1000

# Run the Metropolis-Hastings algorithm
samples = []
likelihood_samples = []
current_sub = X
sampless=[]

for i in range(num_iter):
    # Draw a new proposal from the proposal distribution
    print(i)

    new_sub = TemperedStableSubordinator(0.0, 10.0, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path()
    log_alpha = log_likelihood(new_sub, Y) - log_likelihood(current_sub, Y)
    alpha = np.exp(log_alpha)

    accept = np.random.uniform() < alpha

    if accept:
        current_sub = new_sub

    sampless.append(log_likelihood(new_sub, Y))
    samples.append(current_sub)
    likelihood_samples.append(log_likelihood(current_sub, Y))


print(likelihood_samples)
print(len(likelihood_samples))
print(sampless)
print(len(sampless))
Xs = np.linspace(t1, t2, num_obs)

fig, ax = plt.subplots(nrows=3, figsize=(12,8))
ax[0].plot(Xs, Y, label='initial')
    #ax[0].plot(X, estimates[-1], label='test')
ax[0].set_xlabel('x', fontsize=15)
ax[0].set_ylabel('y(x)', fontsize=15)
ax[0].grid(True)

    #plt.legend()

ax[1].plot(Xs, initial_sub, label='Initial')
#for k in range(5):
#ax[1].plot(X, samples[-1], label='Estimate')
#ax[1].plot(X, samples[-1], label='Estimate')
ax[1].title.set_text('Subordinator')
ax[1].set_xlabel('x', fontsize=15)
ax[1].set_ylabel('W(x)', fontsize=15)
ax[1].grid(True)
ax[1].legend()


ax[2].plot(np.linspace(1, len(likelihood_samples),len(likelihood_samples)), likelihood_samples, label='likelihood')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()
