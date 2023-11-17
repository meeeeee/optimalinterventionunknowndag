import numpy as np
from typing import List, Tuple

"""
data: collection of samples clustered by intervention applied where each element in the list is an np.ndarray with first index sample number
params: prior parameters - a tuple of 4 parameters: alpha_mu, alpha_W, nu, T^-1

Updates the parameters using the data and uses Woodbury if the number of samples is less than the number of nodes
"""
def update_params(data: List[np.ndarray], params: Tuple[float, float, np.ndarray, np.ndarray]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    new_params = [params[0], params[1], params[2], params[3]] # to make typehints happy
    N = sum(samples.shape[0] for samples in data)

    means = [np.mean(samples, 0) for samples in data]
    covs = [np.sum((data[i] - means[i])@(data[i] - means[i]).T, 0) for i in range(len(data))]
    
    #if N > params[2].shape[0]: # invert entire T because it's more efficient than Woodbury; TODO: have not implemented Woodbury
    new_params[3] = np.linalg.inv(new_params[3])

    new_params[3] += sum(covs) + sum([(data[i].shape[0]*new_params[0]/(N + new_params[0]))*((means[i] - new_params[2])@(means[i] - new_params[2]).T) for i in range(len(data))])
    
    new_params[3] = np.linalg.inv(new_params[3])
    #else: # use Woodbury formula to update T^-1
    #    U = np.concat([data[i] - means[i] for i in range(len(data))])

    new_params[2] = (sum([data[i].shape[0]*means[i] for i in range(len(data))]) + new_params[0]*new_params[2])/(N + new_params[0])
    new_params[1] += N
    new_params[0] += N

    return (new_params[0], new_params[1], new_params[2], new_params[3])
