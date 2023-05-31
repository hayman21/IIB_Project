import numpy as np
from Base_Files.ClassLevyJumpProcesses import TemperedStableSubordinator
import matplotlib.pyplot as plt

t1 = 0.0
t2 = 10.0
num_obs = 100
num_epochs = 2000
subordinator_truncation = 0.0
kappa = 0.7
delta = 1.5
gamma = 1.0
nProcesses = 1
l=1

g_sub1 = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=0.7, delta=1.5, gamma=1.0)
g_sub2 = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=0.2, delta=1.2, gamma=0.3)
X = np.linspace(t1, t2, num_obs).reshape(-1, 1)

fig, ax = plt.subplots(nrows=2, figsize=(12,8))

ax[0].plot(X, g_sub1.generate_path().reshape(-1,1))
ax[1].plot(X, g_sub2.generate_path().reshape(-1,1))

plt.show()