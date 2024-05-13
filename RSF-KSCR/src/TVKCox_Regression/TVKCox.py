import numpy as np
import pandas as pd
from scipy.optimize import minimize
from functools import partial
from scipy.stats import norm
from helper_functions import generate_basis_function_list
from time import time, perf_counter
from sklearn.base import BaseEstimator
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from scipy.integrate import quad
import joblib

class ADMM_TVKCox_optimizer:
    """
        An optimizer for the ADMM algorithm used in the TVKCox model. The optimizer is used to find the optimal B-spline
        coefficients for the model given a set of hyperparameters. The optimizer is used in the TVKCox_regression class.

        Attributes
        ----------
        q : int
            The number of B-spline coefficients used in the model.
        m : int
            The degree of the B-splines
        n : int
            The number of unique observations in the input data.
        p : int
            The number of covariates in the input data.
        kernel : callable
            The kernel function used in the model.
        bw : float
            The bandwidth of the kernel function.
        spline_basis_functions : list
            A list of functolls partial basis functions used in the model.
        knots : list
            A list of knots used in the B-spline basis functions.
        covariate_list : list
            A list of covariate names used in the model.
        random_start_variance : float
            The variance of normal distribution generating the random starting points
            for the optimization algorithm
        random_state : int
            The random seed.
        verbose : int
            The verbosity level of the optimizer.
        
        Returns
        -------
        ADMM_TVKCox_optimizer instance
            The optimizer.
    """
    def __init__(self, q: int, m: int, n:int = None, p:int = None, kernel = norm.pdf, bw: float = 10.0, 
                 spline_basis_functions = None, knots:list = None, cov_list:list = None, random_start_variance:float = 0.1,
                 random_state:int = 1, verbose: int = 0):
        """
          Initialize an instance of ADMM_TVKCox_optimizer.
  
          Parameters
          ----------
          q : int
              The number of B-spline coefficients used in the model.
          m : int
              The degree of the B-splines
          n : int
              The number of unique observations in the input data.
          p : int
              The number of covariates in the input data.
          kernel : callable
              The kernel function used in the model.
          bw : float
              The bandwidth of the kernel function.
          spline_basis_functions : list
              A list of functolls partial basis functions used in the model.
          knots : list
              A list of knots used in the B-spline basis functions.
          cov_list : list
              A list of covariate names used in the model.
          pllh_list : list
              A list to hold partial log-likelihood values from the optimization.
          gamma_matrix : np.ndarray
              The B-spline coefficient matrix.
          theta_matrix : np.ndarray
              The slack variable matrix.
          verbose : int
              The verbosity level of the optimizer.
          
          Returns
          -------
          ADMM_TVKCox_optimizer instance
              The optimizer.
        """
        
        np.random.seed(random_state) 
        self.q = q
        self.m = m
        self.n = n
        self.p = p
        self.covariate_list = cov_list
        self.knots = knots
        self.bw = bw
        self.pllh_list = []
        self.kernel = partial(kernel, scale=bw)
        self.spline_basis_functions = spline_basis_functions
        self.gamma_matrix = random_start_variance*np.random.randn(self.p, self.q)
        norms = np.linalg.norm(self.gamma_matrix, axis=1)
        
        # Make sure that the random start variance is not too small for any of the coefficients
        while np.any(norms < 1e-8):
            self.gamma_matrix = random_start_variance*np.random.randn(self.p, self.q)
            norms = np.linalg.norm(self.gamma_matrix, axis=1)
        
        self.theta_matrix = np.zeros((self.p, self.q)) 
        self.verbose = verbose
    
    def _Regular_solver(self, X: pd.DataFrame, nu, scipy_tol):
        """
          Naive solver for the optimization problem without regularization using the the PLLH
          function directly with l1 norm regularization. 
          Only implemented for constant coefficients
  
          Parameters
          ----------
          X : pd.DataFrame
              The input data.
          nu : float
              The regularization parameter.
          scipy_tol : float
              The tolerance of the scipy optimization algorithm.
          
          Returns
          -------
          ADMM_TVKCox_optimizer instance
              The fitted optimizer.
        """
        np.random.seed(87)
        if self.q > 1:
            raise NotImplementedError("Regular solver not implemented for q > 1")
        
        def neg_pllh(gamma_matrix, X: pd.DataFrame, nu):
            term_1 = - self._pllh(gamma_matrix, X)
            term_2 = nu*np.linalg.norm(gamma_matrix, ord=1)
            return term_1 + term_2
        
        opt_print = False
        if self.verbose >= 2:
            opt_print = True

        t0 = perf_counter()
        if scipy_tol is not None:
            gamma = minimize(fun=neg_pllh, x0=self.gamma_matrix.reshape(self.p), args=(X, nu),options={"disp": opt_print}, tol=scipy_tol).x
        else:
            gamma = minimize(fun=neg_pllh, x0=self.gamma_matrix.reshape(self.p), args=(X, nu),options={"disp": opt_print}).x
        self.gamma_matrix = gamma
        t1 = perf_counter()

        if self.verbose >= 1:
            print(f"Finished non-regularized optimization in total time: {np.round(t1-t0, 4)} seconds")
        
        return self
    
    def _StepSizeOptHelper(self, data, gamma, old_gamma_j, new_gamma_j, grad, j, t, success):
        """
          Helper function for the step size optimization in the ADMM algorithm.
          Calculates the criteria for the step size optimization.
  
          Parameters
          ----------
          data : pd.DataFrame
              The input data.
          gamma : np.ndarray
              The B-spline coefficient matrix.
          old_gamma_j : np.ndarray
              The old B-spline coefficient vector for covariate j.
          new_gamma_j : np.ndarray
              The new B-spline coefficient vector for covariate j to be tested.
          grad : np.ndarray
              The gradient of the partial log-likelihood function.
          j : int
              The covariate number.
          t : float
              The step size.
          
          Returns
          -------
          Boolean 
              Indicator whether the criteria is met.
          New gradient
              The new gradient of the partial log-likelihood function.
        """
        
        new_gamma = gamma.copy()
        new_gamma[j,:] = new_gamma_j
        old_gamma = gamma.copy()
        old_gamma[j,:] = old_gamma_j
        
        new_pllh = self._pllh(new_gamma, data)
        new_grad = self._pllh_grad_j(data, new_gamma, j)
        
        old_pllh = self._pllh(old_gamma, data)
        gamma_j_diff = new_gamma_j - old_gamma_j

        return (new_pllh <= old_pllh + grad.dot(gamma_j_diff) + (1/(2*t))*np.linalg.norm(gamma_j_diff)**2), new_grad 

    def _minimize_helper(self, pllh_grad, theta_vector, temp_gamma_matrix, alpha, Lambda, w, j, d, t, pairwise_dist, verbose, opt_print, opt_algorithm, surrogate_likelihood, scipy_tol):
        """
          Helper function for the GroupedDescent_helper and _StepSizeOpt functions. Minimizes the surrogate
          partial log likelihood function function using the chosen algorithm in scipy minimize.
  
          Parameters
          ----------
          pllh_grad : np.ndarray  
              The gradient of the partial log-likelihood function.
          theta_vector : np.ndarray
              The slack variable vector.
          temp_gamma_matrix : np.ndarray
              The current iteration B-spline coefficient matrix.
          alpha : float
              The regularization mixing parameter.
          Lambda : float
              The penalty parameter.
          w : np.ndarray
              The network penalty weights.
          j : int
              The covariate number.
          d : np.ndarray
              The sum of the network penalty weights.
          t : float
              The step size.
          pairwise_dist : np.ndarray
              The pairwise distance matrix for network penatly calculation.
          verbose : int
              The verbosity level of the optimizer.
          opt_print : bool
              Indicator whether to print the optimization process.
          opt_algorithm : str
              The optimization algorithm to use.
          surrogate_likelihood : bool
              Indicator whether to use the surrogate likelihood function.
          scipy_tol : float
              The tolerance of the scipy optimization algorithm.
          
          Returns
          -------
          Scipy optimization result 
              The optimization results
 
        """
        if opt_algorithm == "Nelder-Mead" or opt_algorithm=="BFGS":
                if scipy_tol is not None:
                    opt_res = minimize(fun= self._pllh_surrogate, x0=temp_gamma_matrix[j,:], args=(temp_gamma_matrix, theta_vector, alpha, Lambda, t, j, pllh_grad, w, pairwise_dist, d), method=opt_algorithm, options={"disp": opt_print}, tol=scipy_tol)
                else:
                    opt_res = minimize(fun= self._pllh_surrogate, x0=temp_gamma_matrix[j,:], args=(temp_gamma_matrix, theta_vector, alpha, Lambda, t, j, pllh_grad, w, pairwise_dist, d), method=opt_algorithm, options={"disp": opt_print})
              
        elif opt_algorithm == "Newton-CG":
                if scipy_tol is not None:
                    opt_res = minimize(fun= self._pllh_surrogate, jac = self._pllh_surrogate_grad_j, x0=temp_gamma_matrix[j,:], args=(temp_gamma_matrix, theta_vector, alpha, Lambda, t, j, pllh_grad, w, pairwise_dist, d), options={"disp": opt_print}, tol=scipy_tol)
                else:
                    opt_res = minimize(fun= self._pllh_surrogate, jac= self._pllh_surrogate_grad_j, x0=temp_gamma_matrix[j,:], args=(temp_gamma_matrix, theta_vector, alpha, Lambda, t, j, pllh_grad, w, pairwise_dist, d), options={"disp": opt_print}) 
        
        else:
            raise ValueError(f"{opt_algorithm} is not an implemented optimization algorithm, use Nelder-Mead or BFGS")
        
        return opt_res
    
    def _StepSizeOpt(self, data, gamma, new_gamma_j, grad, j, t, scipy_tol, alpha, Lambda, w, pairwise_dist, d, theta_vector, opt_algorithm, opt_print, success):
        """
          Helper function for the GroupedDescent_helper function, optimizes the step size.
  
          Parameters
          ----------
          data : pd.DataFrame
              The input data.
          gamma : np.ndarray
              The B-spline coefficient matrix.
          new_gamma_j : np.ndarray
              The new B-spline coefficient vector for covariate j.
          grad : np.ndarray
              The gradient of the partial log-likelihood function.
          j : int
              The covariate number.
          t : float
              The step size.
          scipy_tol : float
              The tolerance of the scipy optimization algorithm.
          alpha : float
              The regularization mixing parameter.
          Lambda : float
              The penalty parameter.
          w : np.ndarray
              The network penalty weights.
          pairwise_dist : np.ndarray
              The pairwise distance matrix for network penatly calculation.
          d : np.ndarray
              The sum of the network penalty weights.
          theta_vector : np.ndarray
              The slack variable vector.
          opt_algorithm : str
              The optimization algorithm to use.
          opt_print : bool
              Indicator whether to print the optimization process.
          success : bool
              Indicator whether the scipy minimze optimization was successful.
          
          Returns
          -------
          New_gamma_j : np.ndarray
              The coefficient vector for covariate j.
 
        """
        
        bool_cond = True
        old_gamma_j = gamma[j,:]
        current_grad = grad
        
        # While stepsize is to large to fulfill the likelihood change criteria reduce it's size and find new coefficent vector
        while bool_cond:
            test, current_grad = self._StepSizeOptHelper(data, gamma, old_gamma_j, new_gamma_j, current_grad, j, t, success)
            if test:
                break
            else:
                t = 0.8*t            
                if self.verbose >= 2:
                    print(f"Step size reduced to: {t}")
                if t < 1.0e-8:
                    break
                opt_res = self._minimize_helper(current_grad, theta_vector, gamma, alpha, Lambda, w, j, d, t, pairwise_dist, self.verbose, opt_print, opt_algorithm, True, scipy_tol)
                old_gamma_j = new_gamma_j
                new_gamma_j = opt_res.x
        
        return new_gamma_j
    
    def _GroupedDescent_helper(self, data, neg_pllh, temp_theta_matrix, temp_gamma_matrix, alpha, phi, Lambda, w, j, d, t, pairwise_dist, verbose, opt_print, opt_algorithm, surrogate_likelihood, scipy_tol):
        """
          Helper function for the GroupedDescent_solver. Minimizes the partial log-likelihood function for covariate j.
  
          Parameters
          ----------
          data : pd.DataFrame
              The input data.
          neg_pllh : callable
              The negative partial log-likelihood function.
          temp_theta_matrix : np.ndarray
              The current iteration slack variable matrix.
          temp_gamma_matrix : np.ndarray
              The current iteration B-spline coefficient matrix.
          alpha : float
              The regularization mixing parameter.
          phi : float
              A regularization parameter.
          Lambda : float
              The penalty parameter.
          w : np.ndarray
              The network penalty weights.
          j : int
              The covariate number.
          d : np.ndarray
              The sum of the network penalty weights.
          t : float
              The step size.
          pairwise_dist : np.ndarray
              The pairwise distance matrix for network penatly calculation.
          verbose : int
              The verbosity level of the optimizer.
          opt_print : bool
              Indicator whether to print the optimization process.
          opt_algorithm : str
              The optimization algorithm to use.
          surrogate_likelihood : bool
              Indicator whether to use the surrogate likelihood function.
          scipy_tol : float
              The tolerance of the scipy optimization algorithm.
          
          Returns
          -------
          gamma_j : np.ndarray
              The new coefficient vector for covariate j.
          success : bool
              Indicator whether the scipy minimze optimization was successful.
 
        """
        
        theta_vector = temp_theta_matrix[j,:]
        # Use grouped descent to optimize the surrogate likelihood
        if surrogate_likelihood:
            pllh_grad = self._pllh_grad_j(data, temp_gamma_matrix, j)
            opt_res = self._minimize_helper(pllh_grad, theta_vector, temp_gamma_matrix, alpha, Lambda, w, j, d, t, pairwise_dist, verbose, opt_print, opt_algorithm, surrogate_likelihood, scipy_tol)
            gamma_j = opt_res.x
            gamma_j = self._StepSizeOpt(data, temp_gamma_matrix, gamma_j, pllh_grad, j, t, scipy_tol, alpha, Lambda, w, pairwise_dist, d, theta_vector, opt_algorithm, opt_print, success = opt_res.success)
            success = opt_res.success
        
        # use grouped descent on the PLLH directly  
        else:
            if opt_algorithm == "Nelder-Mead" or "BFGS":
                if scipy_tol is not None:
                    opt_res = minimize(fun= neg_pllh,  x0=temp_gamma_matrix[j,:], args=(temp_gamma_matrix, theta_vector, phi, Lambda, alpha, w, pairwise_dist, j, d, verbose, data), method=opt_algorithm, options={"disp": opt_print}, tol=scipy_tol)
                else:
                    opt_res = minimize(fun= neg_pllh,  x0=temp_gamma_matrix[j,:], args=(temp_gamma_matrix, theta_vector, phi, Lambda, alpha, w, pairwise_dist, j, d, self.verbose, data), method=opt_algorithm, options={"disp": opt_print})
                gamma_j = opt_res.x
                success = opt_res.success
            else:
                raise ValueError(f"{opt_algorithm} is not an implemented optimization algorithm, use Nelder-Mead or BFGS")
        
        return gamma_j, success
    
    def _GroupedDescent_solver(self, X: pd.DataFrame, iter_limit: int, phi: float, Lambda: float, nu: float, scipy_tol:float,
                               alpha:float, tol: float, t: float, net_pen:bool, weight_threshold:float, penalty_type:str = "L0",
                               surrogate_likelihood:bool = True, opt_algorithm:str = "BFGS", random_descent_cycle:bool = True, rand_state:int = 1):
        """
          The main solver function for the ADMM algorithm. The function performs the generalized ADMM algorithm
          introduced in the paper "Time-varying Hazards Model for Incorporating Irregularly Measured High_dimensional 
          Biomarkers" by Xiang et al. (2020).
  
          Parameters
          ----------
          X : pd.DataFrame
              The input data.
          iter_limit : int
              The maximum number of iterations.
          phi : float
              A regularization parameter.
          Lambda : float
              The penalty parameter.
          nu : float
              A regularization parameter.
          scipy_tol : float
              The tolerance of the scipy optimization algorithm.
          alpha : float
              The regularization mixing parameter.
          tol : float
              The convergence tolerance.
          t : float
              The step size.
          net_pen : bool
              Indicator whether to use network penalty.
          weight_threshold : float
              The threshold for the network penalty weights to be non-zero.
          penalty_type : str
              The penalty type to use (L0 or L1).
          surrogate_likelihood : bool
              Indicator whether to use the surrogate likelihood function.
          opt_algorithm : str
              The optimization algorithm to use in the convex optimization step.
          random_descent_cycle : bool
              Indicator whether to use random descent cycle or a fixed one.
          rand_state : int
              The random seed.
           
          Returns
          -------
          ADMM_TVKCox_optimizer instance
              The fitted optimizer.

        """
        
        np.random.seed(rand_state)
        t2_0 = perf_counter()
        
        # Define the negative PLLH do be used instead of surrogate
        def neg_pllh(gamma_vec, gamma_mat, theta_vec, phi, Lambda, alpha, w, pairwise_dist, j, d, verbose,  X: pd.DataFrame):
            gamma_mat[j,:] = gamma_vec
            loglik = (-self._pllh(gamma_matrix=gamma_mat, df=X))
            penalty = alpha*phi*np.sqrt(self.q)*np.linalg.norm(gamma_vec-theta_vec)
            if w is not None:
                network_penalty = np.sum(w[j]*pairwise_dist[j])
                network_penalty = (1-alpha)*(Lambda/2)*np.sqrt(self.q)*network_penalty
            else:
                network_penalty = 0
            if verbose >= 3:
                print(f"loglik: {loglik}, penalty: {penalty}, network penalty: {network_penalty}")
            return  loglik + penalty + network_penalty

        data = X
        opt_print = False
        
        if net_pen:
            # Calculate correlations, threshold to obtain weights
            w = (np.abs(X[self.covariate_list].corr()) - np.eye(X[self.covariate_list].shape[1])).values
            w = np.where(w >= weight_threshold, w, 0)
            d = np.sum(w, axis=0)
            d = np.where(d != 0, d, 1)
            
            if self.verbose >= 3:
                opt_print = True
                print(f"network penalty weights: {w}")
        else:
            w = None
            d = None
        
        # Initiate b-spline coefficient matrix (gamma_matrix) and slack variable matrix (theta_matrix)
        # with a copy to be used when comparing changes between iterations
        temp_gamma_matrix = np.zeros(self.gamma_matrix.shape)
        temp_gamma_matrix[:,:] = self.gamma_matrix
        temp_theta_matrix = np.zeros(self.theta_matrix.shape)
        temp_theta_matrix[:,:] = self.theta_matrix

         #self.pllh_list.append(self._pllh(data, temp_gamma_matrix))
        self.pllh_list.append(-10000)

        # For constant coefficients use regular ADMM with dual variable updates
        if self.q == 1:
            nu = np.ones((self.p,1))*nu

        # Perform the ADMM algorithm
        for i in range(iter_limit):
            t0_1 = perf_counter()
            if net_pen:
                # Calculate the current coefficient norms and norm differences 
                # for the network penalty
                norm_vec = np.linalg.norm(self.gamma_matrix, axis=1) / np.sqrt(d)
                pairwise_dist = np.square(norm_vec[None,:] - norm_vec[:,None])
            else:
                pairwise_dist = None
            
            if i > 0:
                last_val = p_perm[-1]
            
            p_perm = np.random.permutation(self.p)
            
            # make sure we don't cycle to the same coefficient twice in a row due to permutation randomnes.
            while i > 0 and p_perm[0] == last_val and self.p > 1:
                p_perm = np.random.permutation(self.p)

            if self.verbose >= 3:
                print(f"Coefficient matrix at the start of iteration {i}: {self.gamma_matrix}")
            if random_descent_cycle:
                iter = p_perm
            else:
                iter = range(self.p)
            
            for j in iter:
                t0 = perf_counter()
                
                # Perform convex optimization step
                gamma_j, success = self._GroupedDescent_helper(data, neg_pllh, temp_theta_matrix, temp_gamma_matrix, alpha, phi, Lambda, w, j, d, t, pairwise_dist, self.verbose, opt_print, opt_algorithm, surrogate_likelihood, scipy_tol)
                temp_gamma_matrix[j,:] = gamma_j

                if not success and self.verbose >= 3:
                    print(f"Optimization did not converge for covariate num {j} on iteration {i+1}")
                
                # For constant coefficients use regular ADMM with dual variable updates
                if penalty_type != "L1" and penalty_type != "L0":
                    raise ValueError(f"{penalty_type} is not an implemented penalty type, use L1 or L0")
                
                if self.q == 1:
                    if penalty_type == "L1":
                        if np.linalg.norm(gamma_j) == 0:
                            temp_theta_matrix[j,:] = np.zeros(gamma_j.shape)
                        else:
                            temp_theta_matrix[j,:] = int((1 - ((nu[j,:] / phi) / np.linalg.norm(gamma_j))) >= 0)*gamma_j
                    else:
                        temp_theta_matrix[j,:] = gamma_j*int(np.linalg.norm(gamma_j) > nu[j,:] / phi)

                    nu[j,:] += nu[j,:] * phi*(temp_gamma_matrix[j,:] - temp_theta_matrix[j,:])
                else:
                    if penalty_type == "L1":
                        if np.linalg.norm(gamma_j) == 0:
                            temp_theta_matrix[j,:] = np.zeros(gamma_j.shape)
                        else:
                            temp_theta_matrix[j,:] = int((1 - ((nu / phi) / np.linalg.norm(gamma_j))) >= 0)*gamma_j
                    else:
                        temp_theta_matrix[j,:] = gamma_j*int(np.linalg.norm(gamma_j) > nu / phi)

                t1 = perf_counter()
                if self.verbose >= 2:
                    print(f"Time taken for optimization on covarianta num {j}: {np.round(t1-t0, 4)} seconds")
            
            t1_1 = perf_counter()
            if self.verbose >= 1:
                print(f"Time used on iteration: {i+1}: {np.round(t1_1-t0_1, 4)} seconds")
            
            pllh = self._pllh(df=data, gamma_matrix=temp_gamma_matrix)
            if pllh == 0:
                self.pllh_list.append(pllh)
                self.gamma_matrix[:,:] = temp_gamma_matrix

                t2_3 = perf_counter()

                if self.verbose >= 1:
                    print(f"Finished after {i+1} iterations due to pllh == 0 in a total time of {np.round(t2_3 - t2_0, 4)} seconds")
                
                return self
            
            # calculate convergence criteria and check
            relative_pllh_change = abs(pllh - self.pllh_list[-1]) / abs(pllh)
            if self.q == 1:
                if np.linalg.norm(temp_gamma_matrix) == 0:
                    relative_gamma_change = 0
                else:
                    relative_gamma_change = np.linalg.norm(temp_gamma_matrix - self.gamma_matrix) / np.linalg.norm(temp_gamma_matrix)
            else:
                if np.all(np.linalg.norm(temp_gamma_matrix, axis=0) == 0):
                    relative_gamma_change = np.zeros(self.q)
                else:
                    relative_gamma_change = np.linalg.norm(temp_gamma_matrix - self.gamma_matrix, axis=0) / np.linalg.norm(temp_gamma_matrix, axis=0)
            
            if self.verbose >= 1:
                print(f"rel pllh change: {relative_pllh_change}, rel gamma change: {relative_gamma_change}")
            
            if self.q == 1:
                if relative_pllh_change < tol or relative_gamma_change < 0.01:
                    self.pllh_list.append(pllh)
                    self.gamma_matrix[:,:] = temp_gamma_matrix

                    t2_4 = perf_counter()

                    if self.verbose >= 1:
                        print(f"Finished after {i+1} iterations in a total time of {np.round(t2_4-t2_0, 4)} seconds")

                    return self
            else:
                if relative_pllh_change < tol or np.all(relative_gamma_change < 0.01):
                    self.pllh_list.append(pllh)
                    self.gamma_matrix[:,:] = temp_gamma_matrix

                    t2_4 = perf_counter()

                    if self.verbose >= 1:
                        print(f"Finished after {i+1} iterations in a total time of {np.round(t2_4-t2_0, 4)} seconds")

                    return self
            
            # Update parameter values before next iteration
            self.pllh_list.append(pllh)
            self.gamma_matrix[:,:] = temp_gamma_matrix
            self.theta_matrix[:,:] = temp_theta_matrix
            if self.verbose >= 2:
                print(f"Coefficient matrix at the end of iteration {i+1}: {self.gamma_matrix}")
        
        t2_1 = perf_counter()
        if self.verbose >= 1:
            print(f"Optimization finished without convergence below tolerance in total time {np.round(t2_1-t2_0, 4)} seconds")
            
        return self
    

    def _pllh(self, gamma_matrix, df: pd.DataFrame) -> float:
        """
        Computes the partial log-likelihood of the model on the input data.

        Parameters
        ----------
        df : pd.DataFrame
            The input data.
        gamma_matrix : np.ndarray
            The b-spline coeffient matrix used in the model.

        Returns
        -------
        float
            The partial log-likelihood value.
        """
        np.random.seed(88)
        # Find all rows belonging to non-censored individuals and calculate desiered quantities for the calculation
        cases = df[df["delta"] == 1].copy()
        cases = cases.sort_values(by=["T","id"])
        cases["kernel"] = self.kernel(cases["T"].values-cases["t"].values)
        if self.q > 1:
          # Added try-except blocks due to memory fragmentation issues:
            try:
                cases["gamma_Z"] = np.array((np.sum(gamma_matrix.T*self._Z_i(cases[self.covariate_list], cases["T"]), axis=(1,2))).tolist())
            except:
                cases["gamma_Z"] = np.sum(gamma_matrix.T*self._Z_i(cases[self.covariate_list], cases["T"]), axis=(1,2))
        else:
            try:
                cases["gamma_Z"] = np.array((np.dot(self._Z_i(cases[self.covariate_list], cases["T"]), gamma_matrix)).tolist())
            except:
                cases["gamma_Z"] = np.dot(self._Z_i(cases[self.covariate_list], cases["T"]), gamma_matrix)

        # create a copy of the full dataset to be used for calculation against each case row
        data = df.sort_values(by=["T","id"]).copy()
        
        # Find the number of measurements for each case and mark with id and event time T, used to avoid 
        # repeating calculations 
        unique_id_T = cases.groupby(["id","T"]).size().copy().reset_index().rename(columns={0:"count"})
        
        # Since the normalizing factor is the same for all measurements with the same event time only calculate it for each
        # unique event time and expand to a vector containing #unique measurement for each unique ID with that event time *
        # #unique ids that have this event time
        unique_id_T2 = unique_id_T.groupby(["T"])["count"].sum().copy().reset_index().sort_values(by=["T"])
  
        # Define a helper function that performs the calculations in eq. 4 of Xiang et. al (2020)
        def pllh_helper_func(data_temp, row):
            # Distinqush between constant and time varying coefficients in the calculation
            # Then calculate the second term of eq. 4 with a check for ivalid values for the log function
            if self.q > 1:
                # Added try-except blocks due to memory fragmentation issues:
                try:
                    multiplier = (1/self.n) * np.sum(self.kernel(row["T"] - data_temp["t"].values) *
                                np.exp(np.array((np.sum(gamma_matrix.T*self._Z_i(data_temp[self.covariate_list], row["T"]), axis=(1,2)).tolist()))))
                except:
                    multiplier = (1/self.n) * np.sum(self.kernel(row["T"] - data_temp["t"].values) *
                                np.exp(np.sum(gamma_matrix.T*self._Z_i(data_temp[self.covariate_list], row["T"]), axis=(1,2))))
                temp_val = np.select([multiplier < 1.8*10**(-307), multiplier > 1.8*10**307, multiplier == 0.0], [-1000, 1000, 1e-10], default=np.log(multiplier, where=multiplier != 0))
            else:
                try:
                    multiplier = (1/self.n) * np.sum(self.kernel(row["T"] - data_temp["t"]) *
                                np.exp(np.array((np.dot(self._Z_i(data_temp[self.covariate_list], row["T"]), gamma_matrix).T).tolist())))
                except:
                    multiplier = (1/self.n) * np.sum(self.kernel(row["T"] - data_temp["t"]) *
                                np.exp(np.dot(self._Z_i(data_temp[self.covariate_list], row["T"]), gamma_matrix).T))
                
                temp_val = np.select([multiplier < 1.8*10**(-307), multiplier > 1.8*10**307, multiplier == 0.0], [-1000, 1000, 1e-10], default=np.log(multiplier, where=multiplier != 0))
           
            # Return the result multiplied by the number of measurements for the specific case marked with the case id
            return ((np.ones((int(row["count"])))*temp_val))
        
        # Iterate over all unique case id's and event times to calculate second terms of eq. 4 (same for all measurements
        # from the same subject / id)
        data.set_index(["T"], inplace=True)
        idx_count = 0
        pllh_temp_vals = np.zeros(1)
        for index, row in unique_id_T2.iterrows():

            data = data.loc[row["T"]:]
            pllh_temp_vals += np.sum(cases["kernel"].values[idx_count:int(idx_count + row["count"])] * (cases["gamma_Z"].values[idx_count:int(idx_count + row["count"])] - pllh_helper_func(data, row)))
            idx_count += int(row["count"])

        pllh_val = (1/self.n)*pllh_temp_vals[0]
       
        return pllh_val
   
    def _pllh_grad_j(self, df: pd.DataFrame, gamma_matrix, j):
        """       
        Computes the gradient of the partial log-likelihood with respect to a specific gamma matrix row (covariate dimesion).

        This method calculates the gradient of the partial log-likelihood (pllh) with respect to a particular
        gamma matrix row. It is used in the optimization process of the ADMM Time-Varying Kernel Cox
        Regression model.

        Parameters
        ----------
        df : pd.DataFrame
            The input data.
        gamma_matrix : np.ndarray
            The b-spline coefficient matrix used in the model.
        j : int
            The index of the gamma matrix row (covariate dimension) for which the gradient is computed.

        Returns
        -------
        np.ndarray
            The gradient of the pllh with respect to the specified gamma matrix row.
        """
        np.random.seed(89)
        # Create a list of string names for each basis function
        basis_func_col_names = ["bf" + f"{i}" for i in range(gamma_matrix.shape[1])]

        # Find all rows belonging to non-censored individuals and calculate desiered quantities for the calculation
        cases = df[df["delta"] == 1].copy()
        cases = cases.sort_values(by=["T","id"])
        cases["kernel"] = self.kernel(cases["T"].values-cases["t"].values)
        cases[basis_func_col_names] = self._phi_x(cases[self.covariate_list[j]], cases["T"])
        
        # create a copy of the full dataset to be used for calculation against each case row
        data = df.sort_values(by=["T", "id"]).copy()

        # Find the number of measurements for each case and mark with id and event time T, used to avoid 
        # repeating calculations 
        unique_id_T = cases.groupby(["id","T"]).size().copy().reset_index().rename(columns={0:"count"}).sort_values(by=["T", "id"])
        
        # Since the normalizing factor is the same for all measurements with the same event time only calculate it for each
        # unique event time and expand to a vector containing #unique measurement for each unique ID with that event time *
        # #unique ids that have this event time
        unique_id_T2 = unique_id_T.groupby(["T"])["count"].sum().copy().reset_index().sort_values(by=["T"])#.rename(columns={"count":"count_sum"})
                                                                                  
        # Define a helper function that performs the necessary calculations derived from the analytic derivative of eq.4 in Xiang et. al (2020)
        def pllh_grad_helper_func(data_temp, row):
            # Distinqush between constant and time varying coefficients in the calculation
            # Then calculate the second term of the derivative of eq. 4 with a check for ivalid values for the log function
            # Note that self.phi_x returns an array with more than 1 column when q > 1
            if self.q > 1:
                # Added try-except blocks due to memory fragmentation issues:
                try:
                    multiplier = (self.kernel(row["T"] - data_temp["t"].values) *
                        np.exp(np.array((np.sum(gamma_matrix.T*self._Z_i(data_temp[self.covariate_list], row["T"]), axis=(1,2))).tolist())))
                    multiplier = np.select([np.isposinf(multiplier), np.isneginf(multiplier)], [1.8*10**100, -1.8*10**100], default=multiplier)
                    mult_sum = np.sum(multiplier)
                    temp_val = np.array((np.sum( self._phi_x(data_temp[self.covariate_list[j]], row["T"]) * multiplier.reshape((multiplier.shape[0], 1)) , axis = 0)).tolist()) / np.select([np.isposinf(mult_sum), np.isneginf(mult_sum), mult_sum == 0.0], [1.8*10**300, -1.8*10**300, 1e-10], default=mult_sum)
                except:
                    multiplier = (self.kernel(row["T"] - data_temp["t"].values) *
                        np.exp(np.sum(gamma_matrix.T*self._Z_i(data_temp[self.covariate_list], row["T"]), axis=(1,2))))
                    multiplier = np.select([np.isposinf(multiplier), np.isneginf(multiplier)], [1.8*10**100, -1.8*10**100], default=multiplier)
                    mult_sum = np.sum(multiplier)
                    temp_val = np.sum( self._phi_x(data_temp[self.covariate_list[j]], row["T"]) * multiplier.reshape((multiplier.shape[0], 1)) , axis = 0) / np.select([np.isposinf(mult_sum), np.isneginf(mult_sum), mult_sum == 0.0], [1.8*10**300, -1.8*10**300, 1e-10], default=mult_sum)
            else:
                try:
                    multiplier = (self.kernel(row["T"] - data_temp["t"]).reshape((data_temp.shape[0], 1)) * 
                        np.exp(np.array((np.dot(self._Z_i(data_temp[self.covariate_list], row["T"]), gamma_matrix)).tolist())))
                    multiplier = np.select([np.isposinf(multiplier), np.isneginf(multiplier)], [1.8*10**100, -1.8*10**100], default=multiplier)
                    mult_sum = np.sum(multiplier)
                    temp_val = np.array((np.sum( self._phi_x(data_temp[self.covariate_list[j]], row["T"]) * multiplier.reshape((multiplier.shape[0], 1)) , axis = 0)).tolist()) / np.select([np.isposinf(mult_sum), np.isneginf(mult_sum), mult_sum == 0.0], [1.8*10**300, -1.8*10**300, 1e-10], default=mult_sum)
                except:
                    multiplier = (self.kernel(row["T"] - data_temp["t"]).reshape((data_temp.shape[0], 1)) * 
                        np.exp(np.dot(self._Z_i(data_temp[self.covariate_list], row["T"]), gamma_matrix)))
                    multiplier = np.select([np.isposinf(multiplier), np.isneginf(multiplier)], [1.8*10**100, -1.8*10**100], default=multiplier)
                    mult_sum = np.sum(multiplier)
                    temp_val = np.sum( self._phi_x(data_temp[self.covariate_list[j]], row["T"]) * multiplier.reshape((multiplier.shape[0], 1)) , axis = 0) / np.select([np.isposinf(mult_sum), np.isneginf(mult_sum), mult_sum == 0.0], [1.8*10**300, -1.8*10**300, 1e-10], default=mult_sum)
            return ((np.ones((int(row["count"]), self.q))*temp_val))

        # Iterate over all unique case id's and event times to calculate second terms of the derivative of eq. 4 
        # (same for all measurements were the case ( subject have the same event time))
        data.set_index(["T"], inplace=True)
        idx_count = 0
        pllh_temp_vals = np.zeros((1,self.q))
        for index, row in unique_id_T2.iterrows():

            data = data.loc[row["T"]:]
            if self.q > 1:
                pllh_temp_vals += np.sum(cases["kernel"].values[idx_count:int(idx_count + row["count"])].reshape((cases["kernel"].values[idx_count:int(idx_count + row["count"])].shape[0], 1)) * (cases[basis_func_col_names].values[idx_count:int(idx_count + row["count"]),:] - pllh_grad_helper_func(data, row)), axis=0)
            else:    
                pllh_temp_vals += np.sum(cases["kernel"].values[idx_count:int(idx_count + row["count"])].reshape((cases["kernel"].values[idx_count:int(idx_count + row["count"])].shape[0], 1)) * (cases[basis_func_col_names].values[idx_count:int(idx_count + row["count"])] - pllh_grad_helper_func(data, row)), axis=0)
            idx_count += int(row["count"])

        pllh_val = (1/self.n)*pllh_temp_vals.flatten()
 
        return pllh_val
    
    def _pllh_surrogate(self, gamma_vector, gamma_matrix, theta_vector, alpha, Lambda, t, j, pllh_grad, w, pairwise_dist, d) -> float:
        """
          Computes the surrogate loss function for optimization.
          This method calculates the surrogate loss function derived using the majorization-minimization approach
          outlined in Xiang et. al (2020) section 4 and summarized in eq. 17, which is used in the optimization process
          of the ADMM alogrithm.
          Parameters
          ----------
          gamma_vector : np.ndarray
              The b-spline coefficient vector for a specific row (covariate dimension) of the gamma matrix.
          gamma_matrix : np.ndarray
              The b-spline coefficient matrix used in the model.
          theta_vector : np.ndarray
              The slack variable vector for a specific row (covariate dimension) of the theta matrix.
          Lambda : float
              A regularization parameter.
          t : float
              A tuning parameter.
          j : int
              The index of the gamma matrix row for which the surrogate loss is computed.
          pllh_grad : np.ndarray
              The gradient of the partial log-likelihood (pllh).
          w : np.ndarray
              The network penalty weights.
          pairwise_dist : np.ndarray
              The pairwise distances between the network penalty weights.
          d : np.ndarray
              The sum of the network penalty weights for each covariate dimension.
              
          Returns
          -------
          float
              The value of the surrogate loss function for the specified gamma matrix row.
        """

        term_1 = (1/(2*t)) * np.linalg.norm(gamma_vector - (gamma_matrix[j,:] + t*pllh_grad))**2 
        term_2 = alpha*Lambda*np.sqrt(self.q)*np.linalg.norm(gamma_vector - theta_vector)
       
        if w is not None and pairwise_dist is not None:
            network_penalty = np.sum(w[j]*pairwise_dist[j])
            term_3 = (1-alpha)*(Lambda/2)*np.sqrt(self.q)*network_penalty
        else:
            term_3 = 0
        
        return (term_1 + term_2 + term_3)

    def _pllh_surrogate_grad_j(self, gamma_vector, gamma_matrix, theta_vector, alpha, Lambda, t, j, pllh_grad, w, pairwise_dist, d):
        """
          Computes the gradient of the surrogate loss function with respect to a specific gamma matrix row (covariate dimension).
          This method calculates the gradient of the surrogate loss function with respect to a particular
          gamma matrix row. It is derived by analytical derivation of the surrogate loss function (eq. 17 Xiang et. al (2020)
          It is used in the optimization process of the ADMM algorithm.
          Parameters
          ----------
          gamma_vector : np.ndarray
              The b-spline coefficient vector for a specific row (covariate dimension) of the gamma matrix.
          gamma_matrix : np.ndarray
              The b-spline coefficient matrix used in the model.
          theta_vector : np.ndarray
              The slack variable vector for a specific row (covariate dimension) of the theta matrix.
          Lambda : float
              A regularization parameter.
          t : float
              A tuning parameter.
          j : int
              The index of the gamma matrix row for which the surrogate loss is computed.
          pllh_grad : np.ndarray
              The gradient of the partial log-likelihood (pllh).
          w : np.ndarray
              The network penalty weights.
          pairwise_dist : np.ndarray
              The pairwise distances between the network penalty weights.
          d : np.ndarray
              The sum of the network penalty weights for each covariate dimension.
          
          Returns
          -------
          np.ndarray
              The gradient of the surrogate loss with respect to the specified gamma matrix row.
        """

        term_1 = (1/t) * (gamma_vector - (gamma_matrix[j,:] + t*pllh_grad))
        norm_val = np.linalg.norm(gamma_vector - theta_vector)
        if norm_val == 0:
            term_2 = alpha*Lambda * np.sqrt(self.q) * np.sign(gamma_vector)
        else:
            term_2 = alpha*Lambda * np.sqrt(self.q) * ((gamma_vector - theta_vector) / (norm_val))
        
        if w is not None and pairwise_dist is not None:
            term_3 = np.zeros(gamma_vector.shape)
            if np.linalg.norm(gamma_vector) == 0:
                pass
            else:
                for vec_dim in range(self.q):
                    for i in range(w.shape[1]):
                        term_3[vec_dim] += w[j,i] * ( (1 / d[j]) - ( np.linalg.norm(gamma_matrix[i,:]) / (np.sqrt(d[j])*np.sqrt(d[i])*(np.linalg.norm(gamma_vector)))) )
                term_3 = (1-alpha)*Lambda*np.sqrt(self.q)*gamma_vector*term_3
        else:
            term_3 = np.zeros(gamma_vector.shape)
        
        return (term_1 + term_2 + term_3)

    def _pllh_surrogate_hessian(self, gamma_vector, gamma_matrix, theta_vector, alpha, Lambda, t, j, pllh_grad, w, pairwise_dist, d):
        """
          Computes the Hessian matrix of the surrogate loss function.
          This method calculates the Hessian matrix of the surrogate loss function. It is derived by analytical derivation of
          the gradient expression for a specific gamma matrix row based on eq. 17 from Xiang. et al (2020). It can be used in the optimization
          process of the ADMM algorithm
          Parameters
          ----------
          gamma_vector : np.ndarray
              The b-spline coefficient vector for a specific row (covariate dimension) of the gamma matrix.
          gamma_matrix : np.ndarray
              The b-spline coefficient matrix used in the model.
          theta_vector : np.ndarray
              The slack variable vector for a specific row (covariate dimension) of the theta matrix.
          Lambda : float
              A regularization parameter.
          t : float
              A tuning parameter.
          j : int
              The index of the gamma matrix row for which the surrogate loss is computed.
          pllh_grad : np.ndarray
              The gradient of the partial log-likelihood (pllh).
          w : np.ndarray
              The network penalty weights.
          pairwise_dist : np.ndarray
              The pairwise distances between the network penalty weights.
          d : np.ndarray
              The sum of the network penalty weights for each covariate dimension.
  
          Returns
          -------
          np.ndarray
              The Hessian matrix of the surrogate loss with respect to the specified gamma matrix row.
        """
        q = self.q
        H = np.zeros((q, q))
        for l in range(q):
            for i in range(q):
                norm_val = np.linalg.norm(gamma_vector - theta_vector) 
                if i == l:
                    if w is not None and pairwise_dist is not None:
                        network_penalty = 0
                        norm_j = np.linalg.norm(gamma_vector) 
                        if norm_j == 0:
                            norm_j = norm_j + 1e-12
                        for r in range(w.shape[1]):
                            network_penalty += (w[j,r] * ((1 / d[j]) - (np.linalg.norm(gamma_matrix[r,:]) / (np.sqrt(d[j])*np.sqrt(d[r]))) * 
                                                            ( (1 / (norm_j)) - (gamma_vector[i]**2 / (norm_j**3)))))
                    else:
                        network_penalty = 0

                    if norm_val == 0:
                        term = 0
                    else:
                        term = ((norm_val**2 - (gamma_vector[i] - theta_vector[i])**2) / (norm_val**3))
                    H[l, i] = ((1/t) + alpha*Lambda * np.sqrt(q) * term
                                + (1-alpha)*Lambda*np.sqrt(self.q)*network_penalty)
                else:
                    if w is not None and pairwise_dist is not None:
                        network_penalty = 0
             
                        norm_j = np.linalg.norm(gamma_vector) 
                        if norm_j == 0:
                            norm_j = norm_j + 1e-12
                        for r in range(w.shape[1]):
                            network_penalty += (w[j,r] / (np.sqrt(d[j])*np.sqrt(d[r]))) * gamma_vector[i] * gamma_vector[r] * (np.linalg.norm(gamma_matrix[r,:]) / (norm_j**3))
                    else:
                        network_penalty = 0
                    
                    H[l, i] = (-Lambda*np.sqrt(q)*(gamma_vector[i] - theta_vector[i] + 1.0e-12)*(gamma_vector[l] - theta_vector[l] + 1.0e-12)) / ((norm_val+1.0e-12)**3) + (1-alpha)*Lambda*np.sqrt(self.q)*network_penalty

        return H

    def _Z_i(self, X, u: pd.Series):
        """
        Computes the Z matrix for feature transformation.

        This method calculates the Z matrix from section 2.1 used in the partial log likelihood (eq. 4) in Xiang et. al (2020) 

        Parameters
        ----------
        X : pd.Series
            The input covariate data (X_i(t)).
        u : pd.Series
            A series of event times (T_i).

        Returns
        -------
        np.ndarray
            The Z matrix resulting from the feature transformation.
        """
        # Evaluate basis function values at the given event time(s)
        if not isinstance(u, pd.Series):
            basis_vec = np.array([f(u) for f in self.spline_basis_functions])
        else:
            basis_vec = np.array([f(u) for f in self.spline_basis_functions]).T
        X = X.values
        if X.ndim == 1:
            X = X.reshape((1, X.shape[0]))
        # Treat the case for constant coefficent differently from time varying coefficients
        if self.q > 1:
            Z_matrix_vec = []
            # Treat the cases where we provide event time for multiple cases from those were event time for a single case is provided
            if not isinstance(u, pd.Series):
                for i in range(len(self.spline_basis_functions)):
                    Z_matrix_vec.append((basis_vec[i]*X))
                
            else:
                for i in range(len(self.spline_basis_functions)):
                    Z_matrix_vec.append((basis_vec[:,i].reshape(basis_vec.shape[0],1)*X))
            
            Z_matrix_vec_final = Z_matrix_vec[0]
            for i in range(1, len(self.spline_basis_functions)): 
                Z_matrix_vec_final = np.concatenate((Z_matrix_vec_final, Z_matrix_vec[i]), axis=1)

            if X.shape[0] == 1 and isinstance(u, pd.Series):
                Z_matrix_vec_final = Z_matrix_vec_final.reshape(u.shape[0], self.q, self.p)

            else:
                Z_matrix_vec_final = Z_matrix_vec_final.reshape(X.shape[0], self.q, self.p)

            return np.array(Z_matrix_vec_final)

        else:
            return np.array(X*basis_vec)        

    def _phi_x(self, x:pd.Series, T: pd.Series):
        """
        Computes the dot product between the phi_vector (b-spline basis functions vector) and x_ij(t) (the covariate value for
        individual i covariate j at time t). This is usually done as a column operation s.t x represents x_ij for all individuals
        i at all times t for covariate j

        Parameters
        ----------
        x : pd.Series
            The input covariate data.
        T : pd.Series
            The event time(s) for cases.

        Returns
        -------
        np.ndarray
            The dot product phi_vec.T * x_ij(t) for i = 1...n and t's and specific j.
        """
        # Evaluate basis function values at the given event time(s)
        basis_vec = np.array([f(T) for f in self.spline_basis_functions])
        if x.ndim > 0:
            x = x.values.reshape((x.shape[0], 1))

        value = np.array(x*basis_vec.T).astype(np.float64)

        return value

class Kernels():
	@staticmethod
	def Epanechnikov(x, scale):
		return np.array(np.maximum(0.75*(1 - (x/scale)**2), 0)/scale)

	@staticmethod
	def flat(x,scale):
		return np.array(np.ones(len(x)))
        
class TVKCox_regression(BaseEstimator):
    """
      A class representing an implementation of the Time-Varying Kernel Cox proportional hazards model 
      introduced in the paper "Time-varying Hazards Model for Incorporating Irregularly Measured High_dimensional 
      Biomarkers" by Xiang et al. (2020).
  
      This class provides methods for fitting the Time-Varying Kernel Cox Regression model using a generalized ADMM
      method and performing related operations.
  
      Attribute and method names are chosen to corrisond as good as possible with the notation used in the aforementioned
      paper.
  
      Attributes
      ----------
      data : pd.DataFrame or None
          The input data containing features and survival information.
      gamma_matrix : np.ndarray or None
          The b-spline coefficient matrix used in the model.
      spline_basis_functions : list or None
          The b-spline basis functions for coefficient estimation.
      pllh_list : list
          A list to store partial log-likelihood values during training.
      q : int or None
          The number of b-spline basis functions to use.
      m : int or None
          The degree of the b-spline basis functions to use.
      n : int or None
          The number of unique IDs in the input data.
      p : int or None
          The number of covariate columns in the input data.
      covariate_list : list or None
          The list of covariate column names.
      knots : list or None
          The list of knots used by the b-splines.
      optimizer : ADMM_TVKCox_optimizer instance
          The optimizer used in the model.
      tau : int or None
          The time horizon.
      bw : float or None
          The bandwidth parameter for the kernel.
      phi : float or None
          A regularization parameter.
      nu : float or None
          A regularization parameter.
      Lambda : float or None
          A regularization parameter.
      network_penalty : bool
          A flag to indicate whether to use network penalty.
      w_threshold : float or None
          A threshold value for minimum size rquiered for the network penalty
          weights to be non-zero.
      opt_type : str or None
          The optimization type to use (Grouped descent or regular).
      kernel : callable or None
          The kernel function to use.
      t : float or None
          The step size to use.
      alpha : float or None
          The regularization mixing parameter.
      tol : float or None
          The tolerance value for the optimization loop.
      iter_limit : int or None
          The iteration limit for optimization.
      quantile_knots : bool
          A flag to indicate whether to use quantile knots or equidistant knots.
      track_pllh : bool
          A flag to indicate whether to track partial log-likelihood values during training.
      random_start_variance : float
          The variance of the normal distribution generating random start values
          for the optimization algorithm.
      scoring_method : str
          The scoring method to use for evaluation [c_index_indv, c_index_avg, ct_index, cum_dyn_auc_indv, cum_dyn_auc_avg, custom_cv_score, custom_cv_score_sparse].
      verbose : int
          The verbosity level.
      opt_algorithm : str
          The optimization algorithm to use(BFGS, Nelder-Mead).
      penalty_type : str
          The type of penalty to use (L0, L1).
      surrogate_likelihood : bool
          A flag to indicate whether to use the surrogate likelihood.
      random_descent_cycle : bool
          A flag to indicate whether to use random descent cycles.
      scipy_tol : float
          The tolerance value for the scipy optimization sub algorithm.
      return_full_score : bool
          A flag to indicate whether to return the full score vector.
      rand_state : int
          The random seed
      
      
      Methods
      -------
      init()
          Initializes the ADMM Time-Varying Kernel Cox Regression model.
      fit()
          Fits the ADMM Time-Varying Kernel Cox Regression model to the input data.
      predict()
          Predicts the risk scores for the input data.
      score()
          Computes the chosen score metric for some input data
    
    """

    def __init__(self, data = None, gamma_matrix = None, spline_basis_functions = None, pllh_list = [],
                 q = 1, m = 0, n = None, p = None, covariate_list = None, knots = None, optimizer = None,
                 tau = None, bw = 10, phi = 1, nu = 0.1, Lambda = 1, network_penalty = True,
                 w_threshold = 0.3, opt_type = "GroupedDescent", kernel = norm.pdf, t = 1, alpha = 1.0, 
                 tol = 1.0e-3, iter_limit = 50, quantile_knots = False, track_pllh = False, 
                 random_start_variance = 1.0, scoring_method:str = "c_index", 
                 verbose = 0, opt_algorithm:str = "BFGS", penalty_type:str = "L0", surrogate_likelihood:bool = True,
                 random_descent_cycle:bool = True, scipy_tol:float = None, return_full_score: bool = False, 
                 rand_state: int = 1, pllh_saturated:float = 1000) -> None:
        
        super(BaseEstimator, self).__init__()
        self.data = data
        self.gamma_matrix = gamma_matrix
        self.spline_basis_functions = spline_basis_functions
        self.pllh_list = pllh_list
        self.q = q
        self.m = m
        self.n = n
        self.p = p
        self.covariate_list = covariate_list
        self.knots = knots
        self.optimizer = optimizer
        self.tau = tau
        self.bw = bw
        self.phi = phi
        self.nu = nu
        self.Lambda = Lambda
        self.network_penalty = network_penalty
        self.w_threshold = w_threshold
        self.opt_type = opt_type
        self.kernel = kernel
        self.t = t
        self.alpha = alpha
        self.tol = tol
        self.iter_limit = iter_limit
        self.quantile_knots = quantile_knots
        self.track_pllh = track_pllh
        self.random_start_variance = random_start_variance
        self.scoring_method = scoring_method
        self.verbose = verbose
        self.opt_algorithm = opt_algorithm
        self.penalty_type = penalty_type
        self.random_descent_cycle = random_descent_cycle
        self.surrogate_likelihood = surrogate_likelihood
        self.scipy_tol = scipy_tol
        self.return_full_score = return_full_score
        self.rand_state = rand_state
        self.pllh_saturated = pllh_saturated
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
    
    def set_params(self, **params):
        return super().set_params(**params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
          Fits the ADMM Time-Varying Kernel Cox Regression model to the input data using L0 regularization 
          following the the method outlined in Xiang et. al. (2020) sections 2.1, 2.2, 2.3 and 4.
  
          Parameters
          ----------
          X : pd.DataFrame
              The input data containing features.
          y : pd.Series
              The target variable (survival time).
          q : int
              The number of b-spline basis functions.
          m : int
              The degree of the b-spline basis functions.
          tau : int, optional
              The time horizon (default is 120).
          iter_limit : int, optional
              The iteration limit for optimization (default is 300).
          t : float, optional
              A tuning parameter (default is 0.5).
          Lambda : float, optional
              A regularization parameter (default is 0.0).
          phi : float, optional
              A regularization parameter (default is 2).
          nu : float, optional
              A regularization parameter (default is 0.1).
          tol : float, optional
              Tolerance value for the ADMM optimazation loop (default is 1.0e-3).
          track_pllh : bool, optional
              A flag to track partial log-likelihood values during training (default is True).
          kernel : callable, optional
              The kernel function used in the model (default is scipy.stats.norm.pdf).
          bw : float, optional
              The bandwidth parameter for thw kernel (default is 10.0).
          network_penalty : bool, optional
              A flag to indicate whether to use network penalty (default is True).
          w_threshold : float, optional
              A threshold value for minimum size required for the network penalty weights to be non-zero (default is 0.3).
          opt_type : str, optional
              The optimization type to use (GroupedDescent or Regular) (default is "GroupedDescent").
          verbose : int, optional
              The verbosity level (default is 0).
          random_start_variance : float, optional
              The variance of the normal distribution generating random start values for the optimization algorithm (default is 1.0).
          rand_state : int, optional
              The random seed (default is 1).
  
          Returns
          -------
          ADMM_timevarying_kernel_cox_regression
              The fitted model instance.
        """
    
        req_columns = [col for col in X.columns if col in ["id", "t", "T"]]
        if len(req_columns) != 3:
            raise ValueError("X must contain columns named 'id' with unique subject identifications,"
                              "'t' with measurement times, 'T' with unique event or censoring times"
                              "and 'delta' with binary values (1, 0) indicating event or censoring")

        # Join data to one dataframe
        self.data = X
        self.data["delta"] = y.values
     
        if not self.network_penalty:
            self.alpha = 1.0

        # create padded knot list and b-spline basis function list:
        if self.q < 1 or self.m < 0:
            raise ValueError(f"value of q must be greater or equal to 1 but q={self.q} was provided and value fo m must be non-negative but m={self.m} was prvided")
        
        if self.quantile_knots and self.q > 2 and (self.q-self.m) > 1:
            self.knots = np.array([0, self.tau])
            number = self.q - self.m + 1. -1.
            number = 1. / number
            quantile = self.data["T"].quantile(number)
            self.knots = np.insert(self.knots, 1, quantile)
            number += number
            i = 2
            while number < 1.0:
                quantile = self.data["T"].quantile(number)
                self.knots = np.insert(self.knots, i, quantile)
                number += number
                i += 1
        else:    
            self.knots = np.linspace(0, self.tau, self.q - self.m + 1)
        self.knots = np.append(self.knots, np.ones(self.m)*self.knots[-1])
        self.knots = np.append(np.ones(self.m)*self.knots[0], self.knots)
        self.knots = self.knots.tolist()
        self.spline_basis_functions = generate_basis_function_list(self.q, self.m, self.knots, tau = self.tau)

        # load and/or calculate all provided paramters and variabels
        self.n = len(pd.unique(X["id"]))
        self.covariate_list = [col for col in X.columns if col not in ["T", "t", "delta", "id"]]       
        self.p = len(self.covariate_list)

        # Train the model on the provided data
        self.optimizer = ADMM_TVKCox_optimizer(q = self.q, m=self.m, n = self.n, p = self.p, kernel=self.kernel, bw = self.bw, 
                                          spline_basis_functions=self.spline_basis_functions, knots = self.knots, cov_list=self.covariate_list,
                                          random_start_variance=self.random_start_variance, verbose = self.verbose, random_state=self.rand_state)
        if self.opt_type == "GroupedDescent":
            self.optimizer._GroupedDescent_solver(X = self.data,  iter_limit=self.iter_limit, phi= self.phi, nu = self.nu, alpha = self.alpha, 
                                                  Lambda = self.Lambda, tol = self.tol, t = self.t, net_pen = self.network_penalty, 
                                                  weight_threshold=self.w_threshold, penalty_type=self.penalty_type, scipy_tol=self.scipy_tol,
                                                  surrogate_likelihood=self.surrogate_likelihood, opt_algorithm=self.opt_algorithm, 
                                                  random_descent_cycle=self.random_descent_cycle, rand_state=self.rand_state)
        
        elif self.opt_type == "Regular":
            self.optimizer._Regular_solver(self.data, self.phi, scipy_tol=self.scipy_tol)
        
        else:
            self.gamma_matrix = np.zeros((self.p, self.q))
            self.pllh_list = [self.optimizer._pllh(self.gamma_matrix, self.data)]
            print(f"{self.opt_type} is not an implemented optimization type, use GroupedDescent or Regular. Returning a usable zero model")
            return self
        
        self.gamma_matrix = self.optimizer.gamma_matrix

        if self.track_pllh:
            self.pllh_list = self.optimizer.pllh_list
        else:
            self.pllh_list = [self.optimizer.pllh_list[-1]]
        
        return self
    
    def partial_log_likelihood(self, X: pd.DataFrame, y: pd.Series):
        """
          Computes the partial log-likelihood of the fitted model on the input data.
  
          Parameters
          ----------
          df : pd.DataFrame
              The input data.
          gamma_matrix : np.ndarray
              The b-spline coeffient matrix used in the model.
  
          Returns
          -------
          float
              The partial log-likelihood value.
        """
        np.random.seed(88)
        X["delta"] = y
        # Find all rows belonging to non-censored individuals and calculate desiered quantities for the calculation
        cases = X[X["delta"] == 1].copy()
        cases = cases.sort_values(by=["T","id"])
        cases["kernel"] = self.optimizer.kernel(cases["T"].values-cases["t"].values)
        if self.q > 1:
            # Added try-except blocks due to memory fragmentation issues:
            try:
                cases["gamma_Z"] = np.array((np.sum(self.gamma_matrix.T*self.optimizer._Z_i(cases[self.covariate_list], cases["T"]), axis=(1,2))).to_list())
            except:
                cases["gamma_Z"] = np.sum(self.gamma_matrix.T*self.optimizer._Z_i(cases[self.covariate_list], cases["T"]), axis=(1,2))
        else:
            try:
                cases["gamma_Z"] = np.array((np.dot(self.optimizer._Z_i(cases[self.covariate_list], cases["T"]), self.gamma_matrix)).tolist())
            except:
                cases["gamma_Z"] = np.dot(self.optimizer._Z_i(cases[self.covariate_list], cases["T"]), self.gamma_matrix)

        # create a copy of the full dataset to be used for calculation against each case row
        data = X.sort_values(by=["T","id"]).copy()
        
        # Find the number of measurements for each case and mark with id and event time T, used to avoid 
        # repeating calculations 
        unique_id_T = cases.groupby(["id","T"]).size().copy().reset_index().rename(columns={0:"count"})
        
        # Since the normalizing factor is the same for all measurements with the same event time only calculate it for each
        # unique event time and expand to a vector containing #unique measurement for each unique ID with that event time *
        # #unique ids that have this event time
        unique_id_T2 = unique_id_T.groupby(["T"])["count"].sum().copy().reset_index().sort_values(by=["T"])
  
        # Define a helper function that performs the calculations in eq. 4 of Xiang et. al (2020)
        def pllh_helper_func(data_temp, row):
            # Distinqush between constant and time varying coefficients in the calculation
            # Then calculate the second term of eq. 4 with a check for ivalid values for the log function
            if self.q > 1:
                # Added try-except blocks due to memory fragmentation issues:
                try:
                    multiplier = (1/self.n) * np.sum(self.optimizer.kernel(row["T"] - data_temp["t"].values) *
                                np.exp(np.array((np.sum(self.gamma_matrix.T*self.optimizer._Z_i(data_temp[self.covariate_list], row["T"]), axis=(1,2))).tolist())))
                except:
                    multiplier = (1/self.n) * np.sum(self.optimizer.kernel(row["T"] - data_temp["t"].values) *
                            np.exp(np.sum(self.gamma_matrix.T*self.optimizer._Z_i(data_temp[self.covariate_list], row["T"]), axis=(1,2))))
                temp_val = np.select([multiplier < 1.8*10**(-307), multiplier > 1.8*10**307, multiplier == 0.0], [-1000, 1000, 1e-10], default=np.log(multiplier, where=multiplier != 0))
            else:
           
                try:
                    multiplier = (1/self.n) * np.sum(self.optimizer.kernel(row["T"] - data_temp["t"]) *
                                np.exp(np.array((np.dot(self.optimizer._Z_i(data_temp[self.covariate_list], row["T"]), self.gamma_matrix).T)).tolist()))
                except:
                    multiplier = (1/self.n) * np.sum(self.optimizer.kernel(row["T"] - data_temp["t"]) *
                            np.exp(np.dot(self.optimizer._Z_i(data_temp[self.covariate_list], row["T"]), self.gamma_matrix).T))
                temp_val = np.select([multiplier < 1.8*10**(-307), multiplier > 1.8*10**307, multiplier == 0.0], [-1000, 1000, 1e-10], default=np.log(multiplier, where=multiplier != 0))
           
            # Return the result multiplied by the number of measurements for the specific case marked with the case id
            return ((np.ones((int(row["count"])))*temp_val))
        
        # Iterate over all unique case id's and event times to calculate second terms of eq. 4 (same for all measurements
        # from the same subject / id)
        data.set_index(["T"], inplace=True)
        idx_count = 0
        pllh_temp_vals = np.zeros(1)
        for index, row in unique_id_T2.iterrows():
            data = data.loc[row["T"]:]
            
            pllh_temp_vals += np.sum(cases["kernel"].values[idx_count:int(idx_count + row["count"])] * (cases["gamma_Z"].values[idx_count:int(idx_count + row["count"])] - pllh_helper_func(data, row)))
            idx_count += int(row["count"])

        pllh_val = (1/self.n)*pllh_temp_vals[0]
       
        return pllh_val
    
    def predict(self, df: pd.DataFrame):
        """
          Predict the risk scores for a given dataset.
  
          Parameters
          ----------
          df : pd.DataFrame
              The dataset.
  
          Returns
          -------
          predictions : np.ndarray
              The risk scores.
        """

        if  not isinstance(self.gamma_matrix, np.ndarray):
            raise AttributeError(f"Coefficient matrix is None")
        
        if self.q > 1:
            try:
                X = df[self.covariate_list + ["t"]]
            except:
                print(f"Missing time column, or time column has wrong name (name must be t), adding column of t = 0 for all points")
            
            bf_val = np.zeros((100, self.q))
            ti = np.linspace(0,self.tau,100)

            for i in range(self.q):
                bf_val[:,i] = self.spline_basis_functions[i](ti) 

            est_beta = np.zeros((self.p, 100))
            for i in range(self.p):
                est_beta[i,:] = np.dot(self.gamma_matrix[i, :],bf_val.T)
            
            risk_scores = np.zeros((X.shape[0],1))
            for i in range(X.shape[0]):
                # Added try-except blocks due to memory fragmentation issues:
                try:
                    risk_scores[i] = np.exp(np.array((np.dot(X[self.covariate_list].values[i], est_beta[:,int(X["t"].values[i])]).tolist())))
                except:
                    risk_scores[i] = np.exp(np.dot(X[self.covariate_list].values[i], est_beta[:,int(X["t"].values[i])]))
            
        else:
            # Added try-except blocks due to memory fragmentation issues:
            try:
                risk_scores = np.exp(np.array((np.dot(df[self.covariate_list].values, self.gamma_matrix)).tolist()))
            except:
                risk_scores = np.exp(np.dot(df[self.covariate_list].values, self.gamma_matrix))

        return risk_scores

    def score(self, X: pd.DataFrame, y: pd.DataFrame):
        """
          Calculate the chosen scoring metric for a given dataset.
  
          Parameters
          ----------
          X : pd.DataFrame
              The dataset.
          y : pd.DataFrame
              censoring indicators.
  
          Returns
          -------
          score : np.ndarray
              The metric.
        """

        predictions = self.predict(X).reshape(X.shape[0],)
  
        if self.scoring_method == "c_index_indv" or self.scoring_method == "c_index_avg":
            data_temp = X.copy()
            data_temp["pred"] = predictions
            data_temp["delta"] = y.values
            data_temp = data_temp.dropna(subset=["delta"])
            # Transform event times to appropriate form:
            if self.scoring_method == "c_index_avg":
                data_temp = data_temp.groupby("id").mean()[["T", "delta", "pred"]].reset_index()
            else:
                data_temp["T"] = data_temp["T"] - data_temp["t"]
                data_temp.loc[data_temp["T"] < 0, "T"] = 0
            score = concordance_index_censored(data_temp["delta"].values.astype(bool), data_temp["T"].values, data_temp["pred"])[0]

        elif self.scoring_method == "cum_dyn_auc_indv" or self.scoring_method == "cum_dyn_auc_avg" or self.scoring_method == "ct_index":
            
            data_temp = X.copy()
            data_temp["pred"] = predictions
            data_temp["delta"] = y.values
            data_tr = self.data.copy(deep=True)
            
            # Prepare data for CD-AUC calculations:
            if self.scoring_method == "cum_dyn_auc_avg":
                data_temp = data_temp.groupby("id").mean()[["T", "delta", "pred"]].reset_index()
                data_temp.loc[data_temp["T"] < 0, "T"] = 0
                
                data_tr = data_tr.groupby("id").mean()[["T", "delta"]].reset_index()
                data_tr.loc[data_tr["T"] < 0, "T"] = 0
                times = np.percentile(data_temp["T"], np.linspace(0.1, 0.9, 10)).astype(int)
    
            if self.scoring_method == "cum_dyn_auc_indv":
                data_temp["T"] = data_temp["T"] - data_temp["t"]
                data_temp.loc[data_temp["T"] < 0, "T"] = 0
         
                data_tr["T"] = data_tr["T"] - data_tr["t"]
                data_tr.loc[data_tr["T"] < 0, "T"] = 0
                times = np.percentile(data_temp["T"], np.linspace(0.1, 0.9, 10)).astype(int)
            
            y_events = data_tr[data_tr["delta"] == 1]            
            train_min, train_max = y_events["T"].min(), y_events["T"].max()

            data_temp = data_temp[data_temp["T"] < train_max]
            data_temp = data_temp[data_temp["T"] >= train_min]
            y_events = data_temp[data_temp["delta"] == 1]
            times = np.unique(times)

            test_min, test_max = y_events["T"].min(), y_events["T"].max()
            assert (
                train_min <= test_min < test_max < train_max
            ), "time range or test data is not within time range of training data."
            
            times = times[times >= test_min]
            times = times[times < test_max]
            
            y_te = data_temp[["delta", "T"]].copy(deep=True)
            y_te.loc[:, "event"] = (y_te["delta"] == 1)
            y_te = y_te.drop(["delta"], axis=1)
            y_tr = data_tr[["delta", "T"]].copy(deep=True)
            y_tr.loc[:, "event"] = (y_tr["delta"] == 1)
            y_tr = y_tr.drop(["delta"], axis=1)
           
            y_tr = y_tr.dropna()
            y_te = y_te.dropna()
            y_tr = y_tr[["event", "T"]]
            y_te = y_te[["event", "T"]]
       
            y_pred_te = data_temp["pred"].values
            y_te = np.array(y_te.to_records(index=False))
            y_tr = np.array(y_tr.to_records(index=False))
            
            
            if self.scoring_method == "cum_dyn_auc_indv" or self.scoring_method == "cum_dyn_auc_avg":
                
                score = cumulative_dynamic_auc(y_tr,y_te,y_pred_te, times)
                if self.return_full_score:
                    return np.round(score[0], 2)
                else:
                    return np.round(score[1], 2)
            
            else:
                # Calculate c-index
                df_unique = X.groupby("id").first()[["T", "delta"]].reset_index()
                ct_index = np.zeros(25)
                for t in range(25):
                    N_conc = 0
                    N_comp = 0
                    for i in range(y_pred_te.shape[0]):
                        for j in range(y_pred_te.shape[0]):
                            if j == i:
                                continue
                            else:
                                N_comp_ij = int(df_unique.loc[i, "T"] < times[t] and df_unique.loc[i, "delta"] == 1 and df_unique.loc[j, "T"] > df_unique.loc[i, "T"])
                                N_conc_ij = N_comp_ij * int(y_pred_te[i, t] > y_pred_te[j, t])
                                N_comp += N_comp_ij
                                N_conc += N_conc_ij
                    
                    ct_index[t] = N_conc / N_comp
                
                score = ct_index[-1]
            
        elif self.scoring_method == "pllh":
            score = self.partial_log_likelihood(X, y)
            return score
        
        elif self.scoring_method == "deviance":
            score = (-2) * self.partial_log_likelihood(X, y)
          
        elif self.scoring_method == "custom_cv_score" or self.scoring_method == "custom_cv_score_sparse":
            # Reshape data to (num_unique_ids, num_unique_timepoints)
            data_temp = X.copy()
            data_temp["pred"] = predictions
            data_temp["delta"] = y.values
            data_temp["T"] = data_temp["T"] - data_temp["t"]
            data_temp.loc[data_temp["T"] < 0, "T"] = 0
            
            data_tr = self.data.copy()
            data_tr["T"] = data_tr["T"] - data_tr["t"]
            data_tr.loc[data_tr["T"] < 0, "T"] = 0
            
            y_events = data_tr[data_tr["delta"] == 1]            
            train_min, train_max = y_events["T"].min(), y_events["T"].max()

            data_temp = data_temp[data_temp["T"] < train_max]
            data_temp = data_temp[data_temp["T"] >= train_min]

            y_events = data_temp[data_temp["delta"] == 1]
            
            times = np.percentile(data_temp["T"], np.linspace(0.1, 0.9, 10)).astype(int)
            times = np.unique(times)

            test_min, test_max = y_events["T"].min(), y_events["T"].max()
            assert (
                train_min <= test_min < test_max < train_max
            ), "time range or test data is not within time range of training data."
            
            times = times[times >= test_min]
            times = times[times < test_max]

            y_te = data_temp[["delta", "T"]].copy(deep=True)
            y_te.loc[:, "delta"] = y_te["delta"].astype(bool)
            y_tr = data_tr[["delta", "T"]].copy(deep=True)
            
            y_tr.loc[:, "delta"] = y_tr["delta"].astype(bool)
            y_tr = y_tr.dropna()
            y_te = y_te.dropna()
       
            y_pred_te = data_temp["pred"].values
            y_te = np.array(y_te.to_records(index=False))
            y_tr = np.array(y_tr.to_records(index=False))

            if self.scoring_method == "custom_cv_score":
                score = cumulative_dynamic_auc(y_tr,y_te,y_pred_te, times)
                score = np.abs(0.5-score[1]) 
                
                return np.round(score, 3) 
            else:
                score = cumulative_dynamic_auc(y_tr,y_te,y_pred_te, times)
                score = np.abs(0.5-score[1]) 
                adjuster = 1 - (np.sum(np.linalg.norm(self.gamma_matrix, axis=1) > 1e-2) / self.p)
                return np.round(score * adjuster, 5) 
        
        else:
            raise ValueError(f"{self.scoring_method} is not an implemented scoring method, use c_index_indv,, c_index_avg cum_dyn_auc_indv, cum_dyn_auc_avg, or ct_index")

        return np.round(score, 3)
    
