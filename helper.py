import numpy as np
from typing import Callable, Tuple

prev_inv = dict()

"""
n: number of vertices
p: probability of each edge

Makes Erdos-Renyi graph satisfying a random topological order to effectively generate a random graph
"""
def erdag(n: int, p: float) -> np.ndarray:
    adj = np.zeros((n,n))

    invperm = np.random.permutation([i for i in range(n)]) # equivalent to randomly sampling a permutation inverse

    for i in range(n):
        for j in range(i+1,n):
            if np.random.uniform() <= p:
                if invperm[i] < invperm[j]:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1

    return adj

"""
adj: adjacency matrix of dag
edgedistro: assigns weight to an edge according to a chosen distribution
vardistro: assigns weight to a diagonal entry according to a chosen distribution

Makes matrix B by choosing edge weights according to a specified distribution --- the diagonal contains the nodewise variances
"""
def dagparam(adj: np.ndarray, edgedistro: Callable[[None], float] = lambda _: np.random.normal(0,1), vardistro: Callable[[None], float] = lambda _: np.abs(np.random.normal(0,0.1))) -> np.ndarray:
    weights = np.vectorize(edgedistro)(adj)
    vars = np.vectorize(vardistro)(adj)

    return ((weights*adj) + np.diag(np.diag(vars)))

"""
scm: parameters of linear Gaussian model
intrv: shift intervention parameters

Samples interventional data from the provided scm (a single sample)
"""
def sample_intrv(scm: np.ndarray, intrv: np.ndarray) -> np.ndarray:
    # X = BX + a + e -> X = (I-B)^-1(a + e)
    n = scm.shape[0]

    vec = intrv + np.random.normal(np.zeros((n)), np.diag(scm))
    inv = np.linalg.inv(np.eye(n) - (scm - np.diag(np.diag(scm))))

    return inv@vec

"""
scm: parameters of linear Gaussian model

Samples observational data from the provided scm
"""
def sample(scm: np.ndarray) -> np.ndarray:
    return sample_intrv(scm, np.zeros((scm[0].shape[0])))
