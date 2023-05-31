import numpy as np
import matplotlib.pyplot as plt
from Base_Files.ClassLevyJumpProcesses import TemperedStableSubordinator
from Base_Files.Creating_the_NGP import GaussianProcess

t1 = 0.0
t2 = 10.0
num_obs = 500
num_epochs = 2000
subordinator_truncation = 0.0
kappa = 0.7
delta = 1.5
gamma = 1.0
nProcesses = 1

g_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma)
for i in range(nProcesses):
    g_sub.generate_path()

#plt.plot(np.linspace(t1, t2, num_obs).reshape(-1,1), g_sub.generate_path())
#plt.show()

N = num_obs
W = g_sub.generate_path().reshape(-1, 1)
l = 1

#plt.plot(np.linspace(t1, t2, num_obs).reshape(-1, 1), GaussianProcess(N, coords, l))
##plt.plot(coords, u)
#plt.show()

X = np.linspace(t1, t2, num_obs).reshape(-1, 1)

fig, ax = plt.subplots(nrows=2, figsize=(12,8))

ax[0].plot(X, GaussianProcess(W, l, N))
#ax[0].scatter(obseved_inputs, observed_samples, label='observed samples', c="#D95319")
ax[0].set_xlabel('x', fontsize=15)
ax[0].set_title('Non-Gaussian Process')
ax[0].set_ylabel('y(x)', fontsize=15)
ax[0].grid(True)
ax[0].legend(fontsize=15)

ax[1].plot(X, W, color='green')
ax[1].set_title('Tempered Stable Subordinator')
ax[1].set_xlabel('x', fontsize=15)
ax[1].set_ylabel('W(x)', fontsize=15)
ax[1].grid(True)

plt.tight_layout()
plt.show()