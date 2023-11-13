from helper import *

n = 5
adj = erdag(n, 0.5)
scm = dagparam(adj)
data = sample(scm)
intrv = np.zeros((n))
intrv[-1] = 1
i_data = sample_intrv(scm, intrv)

print("Adjacency matrix:", adj)
print("SCM parameters:", scm)
print("Observational sample:", data)
print("Interventional sample:", i_data, intrv)
