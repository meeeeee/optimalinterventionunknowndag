from helper import *
from model import Model
import numpy as np

n = 2
adj = erdag(n, 0.0) # increase for edges
scm = dagparam(adj) # parameters are edge weights with vertex variances on the diagonal (we assume Bii = 0)
data = sample(scm)
intrv = np.zeros((n))
#intrv[0] = 2
i_data = sample_intrv(scm, intrv) # sampling & intervention stuff seems to work correctly

print("Adjacency matrix:", adj)
print("SCM parameters:\n", scm)
print("Observational sample:", data, data.shape)
print("Interventional sample:", i_data, intrv)

initial_nu, initial_T_inv = np.zeros(n), np.eye(n) # initial guess for nu & T_inv
#initial_T_inv += np.eye(n)
#initial_T_inv = initial_T_inv@initial_T_inv.T#(np.eye(n) - scm + np.diag(np.diag(scm)))@np.diag(np.diag(scm))@(np.eye(n) - scm + np.diag(np.diag(scm))).T
print("initial:\n", initial_nu, "\n", initial_T_inv)
model = Model(n,n,initial_nu, initial_T_inv)
its, samples = 500, 1000
for it in range(its): # it's possible that even with the correct D, recovering T is incorrect and I need to figure out why
    model.update([np.array([sample(scm) for _ in range(samples)])], [intrv for _ in range(1)], it*samples)

T_inv = model.T_inv

print(T_inv, model.getLGM(its*samples), scm) # B is correct

# NOTE: the assumption that D can be obtained from the diagonal is incorrect (and is only apparently true for the case where there are no edges)
# NOTE: I assume for now that the bug is caused by bad code and not by a mistake in my math
# NOTE: the resulting stimate for T_inv has the same value along the main diagonal, which is likely caused by some problem in the code
# TODO: check that I zero out the main diagonal for all relevant calculations or something
# NOTE: big hint that something is wrong --- even with just one vertex, the converged value is incorrect; I will try simplifying the equations for the case where there is only one vertex and checking things from here
