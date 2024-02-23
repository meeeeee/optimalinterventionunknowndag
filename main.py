from helper import *
from model import Model
import numpy as np

n = 2
adj = erdag(n, 0.7) # increase for edges
scm = dagparam(adj) # parameters are edge weights with vertex variances on the diagonal (we assume Bii = 0)
data = sample(scm)
intrv = np.zeros((n))
intrv[0] = 2
i_data = sample_intrv(scm, intrv) # sampling & intervention stuff seems to work correctly

print("Adjacency matrix:", adj)
print("SCM parameters:\n", scm)
print("Observational sample:", i_data)
"""
#print("Interventional sample:", i_data, intrv)

initial_nu, initial_T_inv = np.random.standard_normal(n), np.random.standard_normal((n,n)) # initial guess for nu & T_inv
initial_T_inv += np.eye(n)
initial_T_inv = initial_T_inv@initial_T_inv.T#(np.eye(n) - scm + np.diag(np.diag(scm)))@np.diag(np.diag(scm))@(np.eye(n) - scm + np.diag(np.diag(scm))).T
print("initial:\n", initial_T_inv)
model = Model(n,n,initial_nu, initial_T_inv)
for _ in range(1000): # it's possible that even with the correct D, recovering T is incorrect and I need to figure out why
    model.update([sample(scm)[np.newaxis,:]], [intrv])

T_inv = model.T_inv

print(T_inv, scm)
"""

# NOTE: the assumption that D can be obtained from the diagonal is incorrect (and is only apparently true for the case where there are no edges)
# NOTE: I assume for now that the bug is caused by bad code and not by a mistake in my math
