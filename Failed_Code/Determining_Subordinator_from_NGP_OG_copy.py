import numpy as np
import matplotlib.pyplot as plt
from Base_Files.ClassLevyJumpProcesses import TemperedStableSubordinator
import GPy
from Base_Files.Creating_the_NGP import GaussianProcess

# Set Parameters
t1 = 0.0
t2 = 10.0
num_obs = 500 # (N) number of points e.g. size of data set
num_epochs = 2000
subordinator_truncation = 0.0
kappa = 0.7#0.2
delta = 1.5#1.2
gamma = 1.0#0.3
nProcesses = 1
l = 1

initial_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path().reshape(-1,1)
X = initial_sub
Y = GaussianProcess(initial_sub, 1, num_obs).reshape(-1,1)

def log_likelihood(X, Y):
    kernel = GPy.kern.RBF(input_dim=1)
    sub_kernel = GPy.kern.Brownian(input_dim=1)
    model = GPy.models.GPRegression(X, Y, kernel*sub_kernel)
    log_likelihood = model.log_likelihood()
    return log_likelihood

num_iter = 10000
burn_in = 0

# Run the Metropolis-Hastings algorithm
current_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path().reshape(-1,1)
acceptances = 0

accept_sub_samples = []

likelihood_samples = []
accept_likelihood_samples = []

alphas = []
accept_alphas = []


for i in range(num_iter):
    # Draw a new proposal from the proposal distribution
    print('Iteration:', i+1)

    new_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path().reshape(-1,1)
    log_alpha = log_likelihood(new_sub, Y) - log_likelihood(current_sub, Y)
    alpha = np.exp(log_alpha)

    accept = np.random.uniform() < alpha
    alphas.append(alpha)

    if accept:
        current_sub = new_sub

        acceptances += 1

        accept_likelihood_samples.append(log_likelihood(current_sub, Y))

        accept_alphas.append(alpha)

        accept_sub_samples.append(current_sub)

    likelihood_samples.append(log_likelihood(current_sub, Y))

# Print the acceptance rate
#print('Acceptance rate:', acceptances*100 / num_iter, '%')
print('Number of accepted likelihood samples:', acceptances)
print('Accepted Likelihood samples:', accept_likelihood_samples)
print('Accepted Alphas:', accept_alphas)
print(log_likelihood(X, Y))

Xs = np.linspace(t1, t2, num_obs)

plt.figure(1).set_figwidth(12)
plt.xlabel('x', fontsize=15)
plt.ylabel('y(x)', fontsize=15)
plt.title('Time-Changed Gaussian Process')
plt.plot(Xs, Y)

#fig1, ax = plt.subplots(nrows=2, figsize=(12,8))

#ax[0].plot(Xs, Y)
    #ax[0].plot(X, estimates[-1], label='test')
#ax[0].set_xlabel('x', fontsize=15)
#ax[0].set_ylabel('y(x)', fontsize=15)
#ax[0].grid(True)
plt.figure(2).set_figwidth(12)
plt.plot(Xs, initial_sub, label='Initial')
plt.plot(Xs, accept_sub_samples[-1], label='Test -1')
plt.plot(Xs, accept_sub_samples[-2], label='Test -2')
plt.plot(Xs, accept_sub_samples[-3], label='Test -3')
plt.title('Subordinator')
plt.xlabel('x', fontsize=15)
plt.ylabel('W(x)', fontsize=15)
plt.figure(2).legend()

plt.figure(3).set_figwidth(12)
plt.plot(np.linspace(1, len(likelihood_samples), len(likelihood_samples)), likelihood_samples, label='likelihood')
plt.axhline(log_likelihood(X, Y))
#plt.plot(np.linspace(1, len(alphas), len(alphas)), alphas)
plt.title('Likelihood as a function of iterations')
plt.xlabel('Iteration', fontsize=15)
plt.ylabel('Likelihood', fontsize=15)
plt.figure(3).legend()

plt.figure(4).set_figwidth(12)
plt.plot(np.linspace(1, len(alphas), len(alphas)), alphas)
plt.title('Alphas over iterations')

#plt.figure(5).set_figwidth(12)
#plt.plot(np.linspace(1, len(accept_alphas), len(accept_alphas)), accept_alphas)
#plt.title('Accepted alphas over iterations')

#ax[1].plot(Xs, initial_sub, label='Initial')
#for k in range(5):
#ax[1].plot(X, samples[-1], label='Estimate')
#ax[1].plot(X, samples[-1], label='Estimate')
#ax[1].title.set_text('Subordinator')
#ax[1].set_xlabel('x', fontsize=15)
#ax[1].set_ylabel('W(x)', fontsize=15)
#ax[1].grid(True)
#ax[1].legend()

plt.show()
