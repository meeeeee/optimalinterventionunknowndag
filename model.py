import numpy as np
from typing import Tuple, List, Text
from iBGE_update import update_params

class Model:
    alpha_mu: float
    alpha_W: float
    nu: np.ndarray
    T_inv: np.ndarray

    """
    Initializes parameters of model --- requires that T_inv is invertible and symmetric for it to be well-defined
    """
    def __init__(self, alpha_mu: float, alpha_W: float, nu: np.ndarray, T_inv: np.ndarray):
        assert np.linalg.det(T_inv) != 0 and np.allclose(T_inv, T_inv.T, rtol=1e-5, atol=1e-8)

        self.alpha_mu, self.alpha_W = alpha_mu, alpha_W
        self.nu, self.T_inv = nu, T_inv

    """
    data: list of samples of interventional data --- each element is a sequence of samples
    intrvs: list of interventions applied to each set of samples
    n: number of datapoints (needed for accurate estimate of covariances)

    Updates model parameters (LGM MAP) given observational and/or interventional samples --- note that as long as 
    """
    # TODO: rewrite update to find best intervention, generate data using this intervention, and update the parameters with this data
    def update(self, data: List[np.ndarray], intrvs: List[np.ndarray], n: int):
        assert min([sample.shape[0] > 0 for sample in data]) # check that there are no empty interventions
        

        nu, T, _, B = self.nu[:], np.linalg.inv(self.T_inv), *self.getLGM(n) # prior parameters
        N = sum(samples.shape[0] for samples in data) # number of samples over all interventional data
        IB = np.linalg.inv(np.eye(B.shape[0]) - B)
        alpha_mu, alpha_W = self.alpha_mu + N, self.alpha_W + N

        means = [np.mean(data[i], 0) - IB@intrvs[i] for i in range(len(data))] # X^{j,I}'
        covs = [(data[i] - IB@intrvs[i] - means[i]).T@(data[i] - IB@intrvs[i] - means[i]) for i in range(len(data))] # S_{N_I}
        
        #if N > params[2].shape[0]: # invert entire T because it's more efficient than Woodbury; TODO: have not implemented Woodbury
        
        T += sum(covs) + sum([(data[i].shape[0]*self.alpha_mu)*((means[i] - nu).T@(means[i] - nu)) for i in range(len(data))])/alpha_mu
       
        T_inv = np.linalg.inv(T)
        #else: # use Woodbury formula to update T^-1
        #    U = np.concat([data[i] - means[i] for i in range(len(data))])

        # TODO: RESUME CHECK FROM HERE
        nu = (sum([data[i].shape[0]*means[i] for i in range(len(data))]) + self.alpha_mu*self.nu)/alpha_mu

        self.alpha_mu, self.alpha_W, self.nu, self.T_inv = alpha_mu, alpha_W, nu, T_inv

    """
    n: number of samples

    Returns estimate of D, B from parameter T^-1
    """
    def getLGM(self, n) -> Tuple[np.ndarray, np.ndarray]: # TODO:TODO:TODO: CHECK THAT THIS ESTIMATION WORKS
        # T_inv = (I-B)D(I-B)^T (D is inverse variances) as noted in kuipers & moffa
        # it should be true that D has estimates for inverse variances
        #self.T_inv *= max(1,np.sqrt(n))
        D = np.diag(self.T_inv)

        # eigendecomposition
        L, Q = np.linalg.eigh(self.T_inv)
        print(D, L, Q, self.T_inv)
        
        B = np.eye(self.T_inv.shape[0]) - Q@np.diag(np.sqrt(L/D))

        # Cholesky decomposition
        #L = np.linalg.cholesky(self.T_inv)
        #N = np.linalg.norm(L, axis=1)
        #D = np.diag(N**2)@D
        #L = L@np.diag(1/N)
        #
        #B = np.eye(self.T_inv.shape[0]) - L@np.diag(1/np.sqrt(D))
        
        #self.T_inv *= 1/max(1,np.sqrt(n))
        return (1/np.sqrt(max(1,n)*D), B) # did not previously account for the number of samples and iterations when calculating the covariance

    """
    name: parameter name

    Getter for parameters
    """
    def __getitem__(self, name: Text):
        match name:
            case "alpha_mu":
                return self.alpha_mu
            case "alpha_W":
                return self.alpha_W
            case "nu":
                return self.nu
            case "T_inv":
                return self.T_inv
            case _:
                raise Exception("Invalid parameter name")
