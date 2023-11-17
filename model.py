import numpy as np
from typing import List, Text
from iBGE_update import update_params

class Model:
    alpha_mu: float
    alpha_W: float
    nu: np.ndarray
    T_inv: np.ndarray

    """
    Initializes parameters of model
    """
    def __init__(self, alpha_mu: float, alpha_W: float, nu: np.ndarray, T_inv: np.ndarray):
        self.alpha_mu, self.alpha_W = alpha_mu, alpha_W
        self.nu, self.T_inv = nu, T_inv

    """
    data: list of samples of interventional data

    Updates model parameters (LGM MAP) given observational and/or interventional samples
    """
    # TODO: rewrite update to find best intervention, generate data using this intervention, and update the parameters with this data
    def update(self, data: List[np.ndarray]):
        self.alpha_mu, self.alpha_W, self.nu, self.T_inv = update_params(data, (self.alpha_mu, self.alpha_W, self.nu, self.T_inv))

    """
    name: parameter name

    Get parameter
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
                raise Exception("Incorrect name provided when fetching parameter")
