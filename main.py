from helper import *

n = 5
adj = erdag(5, 0.3)
scm = dagparam(adj)
data = sample(scm)
intrv = np.ones((n))
i_data = sample_intrv(scm, intrv)

print("Adjacency matrix:", adj)
print("SCM parameters:", scm)
print("Observational sample:", data)
print("Interventional sample:", i_data)
