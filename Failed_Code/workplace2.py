import numpy as np
from Base_Files.ClassLevyJumpProcesses import TemperedStableSubordinator
from Base_Files.Creating_the_NGP import GaussianProcess

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

initial_sub = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path().reshape(-1,1)
nx = TemperedStableSubordinator(t1, t2, num_obs, num_epochs, subordinator_truncation, kappa=kappa, delta=delta, gamma=gamma).generate_path().reshape(-1,1)
x = np.linspace(t1, t2, num_obs)
Y3 = GaussianProcess(initial_sub, 1, num_obs).reshape(-1,1)

Y = initial_sub
Y1 = initial_sub[:100]

Z = [0, 0.21, 1.2 ,0.3, 1.4, 0.55, 1.6, 7, 8, 9]

accept_new_alphas = []
accept_new_alphas2 = []

for i in range(len(Z)):

    new_alpha= min(1, Z[i])

    accept2 = new_alpha > 0

    if accept2:
        if new_alpha == 1:
            new_alpha = accept_new_alphas[-1]
            accept_new_alphas.append(new_alpha)
        else:
            accept_new_alphas.append(new_alpha)

print(accept_new_alphas)
print(accept_new_alphas2)
