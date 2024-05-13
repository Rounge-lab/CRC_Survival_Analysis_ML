from functools import partial
import pandas as pd

def phi_l_k(t: float, i: int, k: int, knots: list, tau:int) -> float: 
    discrete_time = knots
    if k == 0:
        bool = (t >= discrete_time[i]) & (t <= discrete_time[i+1]) & (i < len(discrete_time))
        bool = pd.Series(bool)
        return bool.astype(int) 

    coef_1 = (discrete_time[i+k] - discrete_time[i])
    if coef_1 == 0:
        term_1 = 0
    else:
        term_1 = ((t - discrete_time[i]) / (discrete_time[i+k] - discrete_time[i])) * phi_l_k(t, i, k-1, knots=knots, tau=tau)
    
    coef_2 = (discrete_time[i+k+1] - discrete_time[i+1])
    if coef_2 == 0:
        term_2 = 0
    else:
        term_2 = ((discrete_time[i+k+1] - t) / (discrete_time[i+k+1] - discrete_time[i+1])) * phi_l_k(t, i+1, k-1, knots=knots, tau=tau)


    return term_1 + term_2

def generate_basis_function_list(q, m, knots, tau):
  """
        Generates B-spline basis function with a given degree and number
        of knots over a desiered timeframe (assumed to start at 0) using
        the Cox De-Boor formula.

        Parameters
        ----------
        q : int
            The number of B-spline coefficients.
        m : int
            The B-spline order
        knots: list
            A list of knots to use.
        tau: float
            The end of the interval on which the B-splines are calculated

        Returns
        -------
        list[parital_functions]
            A list of functolls partial functions, one for each basis function.
    """
    phi_list = []
    for l in range(0,q):
        partial_phi = partial(phi_l_k, i = l,k = m, knots=knots, tau=tau) 
        phi_list.append(partial_phi)
    return phi_list
