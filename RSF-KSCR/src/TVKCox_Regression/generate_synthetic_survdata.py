# Author Gabriel Balanan
# Modified and extended by: Emil Jettli 

import numpy as np
import pandas as pd
import warnings
from scipy import stats
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

RAND_SEED = 45

class SyntheticSurvOutcomes():
	"""
	Implementation of the random baseline hazard method of
	Harden and Kropko 2018

	k = number of points in baseline hazard function
	T = end time by which no-one survives
	beta = true log hazard ratio values
	N = number of individuals
	"""

	def __init__(self, k = 10, T = 100, p_censored = 0.3):
		self.k = k
		self.T = T
		self.p_censored = p_censored

	def __call__(self, X, beta = np.ones(2), rand_state = RAND_SEED, random_baseline_surv = True, ncc:bool = False):
		N = X.shape[0]

		if random_baseline_surv:
			baseline_surv = self._get_baseline_surv(rand_state, ncc=ncc)
		else:
			baseline_surv = lambda t: (t_times.max() - t)/t_times.max() 

		#Step 2 Make individual survival functions
		t_times = np.arange(self.T +1)

		st = list(map(baseline_surv, t_times))

		ind_surv_discrete = self._make_individual_surv(baseline_surv,
													   t_times,
													   X,
													   beta)
		#Use inverse sampling to generate synthetic survival times
		U_surv = stats.uniform.rvs(0,
								   size = N,
								   random_state = rand_state + 2)

		fu_times = np.array([t_times[ind_surv_discrete[i] >= U_surv[i]].max(initial = 0) for i in range(N)])
		
		return self._random_censoring(fu_times, rand_state + 3, ncc)
	
	def _make_individual_surv(self, baseline_surv, t_times, X, beta):
		mu = X @ beta
		ind_surv = lambda t, mu: baseline_surv(t)**np.exp(mu)
		return ind_surv(t_times, mu[:, np.newaxis])

	def _get_baseline_surv(self, rand_state, ncc:bool = False):
		
		#Step 1 make a random baseline survival function
		k_x = np.hstack([[0, self.T],
						 stats.uniform.rvs(0, 
						 				   scale = self.T,
						 				   size = self.k -2,
						 				   random_state = rand_state)])
		scale = 1.0
		if ncc:
			scale = 0.1
		
		k_y = np.hstack([[0, scale],
						 stats.uniform.rvs(0,
						 				   scale = scale,
						 				   size = self.k -2,
						 				   random_state = rand_state +1)])

		k_x = np.sort(k_x)
		k_y = np.sort(k_y)

		cubic_failCDF = PchipInterpolator(k_x, k_y)

		baseline_surv = lambda t: 1 - cubic_failCDF(t)
	
		return baseline_surv

	def _random_censoring(self, fu_times, rand_state, ncc:bool = False):
		#Create random censoring
		
		N = len(fu_times)
		if ncc:
			event_occured = np.ones(N)
			event_occured[fu_times >= self.T-1] = 0
			
			#Randomly shorten follow-up times in the censored data points
			fu_times[event_occured == 0] = (fu_times.astype(float)[event_occured == 0]*stats.uniform.rvs(0,
																			1,
																			size = (event_occured == 0).sum(),
																			random_state = rand_state + 4))
		
		else:
			event_occured = stats.binom.rvs(1, 
											1 - self.p_censored,
											size = N,
											random_state = rand_state)
		
			#Randomly shorten follow-up times in the censored data points
			fu_times[event_occured == 0] = (fu_times.astype(float)[event_occured == 0]*stats.uniform.rvs(0,
																			1,
																			size = (event_occured == 0).sum(),
																			random_state = rand_state + 4))

		return fu_times.astype(int), event_occured.astype(int)

def get_testdata(N = 100, p_censored = 0.3, beta = np.log([4.0, 0.25]), random_state = RAND_SEED):
	p = len(beta)
	X = stats.norm.rvs(loc = 0.0, 
					   scale = 0.2,
					   size = N*p,
					   random_state = RAND_SEED).reshape(N, p)
	
	outcomegen = SyntheticSurvOutcomes(T = 2*N,
									   p_censored = p_censored)
	fu_times, event_occured = outcomegen(X, beta, RAND_SEED + 1, 
									     random_baseline_surv = True)

	columnnames = ["time", "event"] + ["X" + str(i) for i in range(1, p +1)]
	survdata = pd.DataFrame(np.hstack([fu_times[:,np.newaxis],
										 event_occured[:,np.newaxis], X]), 
										 columns = columnnames)
	if p > len(beta):
		beta = np.hstack([beta, np.zeros(p - len(beta))])
	return survdata

class TimeVaryingSyntheticSurvOutcomes(SyntheticSurvOutcomes):
	def _make_individual_surv(self, baseline_surv, t_times, X, beta):
		#Get the baseline hazard function
		baseline_surv_arr = np.array([baseline_surv(t) for t in t_times])
		
		#Note infinite hazard at endpoint is removed. No one survives the end of this study.
		baseline_hazard = (-np.gradient(baseline_surv_arr[:-1])/baseline_surv_arr[:-1])
		#baseline_hazard = (2/self.T)*np.ones(baseline_hazard.shape[0])
		if beta.ndim > 1:
			systemic_hazard = np.exp(np.sum(X * beta[:,:, np.newaxis].transpose([2,0,1]), axis = 2)).T	
		else:
			systemic_hazard = np.exp(X @ beta).T
	
		#Grid spacing is 1 so can use cumsum to integrate
		integrated_hazards = np.cumsum((baseline_hazard[:,np.newaxis] * systemic_hazard), axis = 0).T
		ind_surv = np.exp(-integrated_hazards)
		return np.vstack([ind_surv.T, np.zeros(X.shape[0])]).T

def get_longitudinal_testdata(N = 100,
							  T = 100,
							  p_censored = 0.3,
							  HR_true = np.array([4.0, 0.25]),
							  time_varying = True,
							  sparse = True,
							  sparse_mean_num_measurements = 4,
							  randstate = 0,
							  random_baseline_surv = True,
							  ncc:bool = False,
							  data_generation_params: dict = {"dt": 1, "tau": 0.5, "mu": 0.4, "sigma": 1, "n_batch": 1},
							  corr_data_generation_params: dict = {"dt": 1, "tau": np.eye(2)*0.5, "mu": np.ones(2)*0.4, "sigma": np.eye(2), "n_dim": 2, "n_batch": 1},
							  discrete_cov: dict = {}):
	""" Longitudinal version of the method of Harden and Kropko 2018"""
	np.random.seed(randstate)
	check_single_data_gen_params(data_generation_params)
	check_corr_data_gen_params(corr_data_generation_params)
	
	beta = np.log(HR_true)
	if beta.ndim > 1:
		num_covariates = beta.shape[1]
	else:
		num_covariates = len(beta)

	it_num_1 = data_generation_params["n_batch"]
	it_num_2 = corr_data_generation_params["n_batch"]
	it_num_3 = len(discrete_cov)
	num_discrete_cov_vals = 0
	for value in discrete_cov.values():
		length = len(value[0])
		if length == 2:
			num_discrete_cov_vals += 1
		else:
			num_discrete_cov_vals += len(value[0])

	if it_num_1 + it_num_2*corr_data_generation_params["n_dim"] + num_discrete_cov_vals != num_covariates:
		raise ValueError(f"""The sum of n_batch for corr_data_generation_params, 
				   			 n_batch_ for data_generation_params and num_discrete_cov_vals 
							 must equal the number of coefficients in beta but #coef in
				   			 beta {num_covariates} while the sum is: 
							 {it_num_1 + it_num_2*corr_data_generation_params["n_dim"] + num_discrete_cov_vals}""")
	
	#Step 1 generate a stocastic process for the predictor variables
	if time_varying:
		Xdata = [simulate_ornstein_uhlenbeck(N = N,
											T = T,
											dt = data_generation_params["dt"],
											tau = data_generation_params["tau"]*T,
											mu = data_generation_params["mu"],
											sigma = data_generation_params["sigma"],
											n_dim = 1,
											randstate = randstate + i) for i in range(it_num_1)]
		X = np.array(Xdata).transpose([1,2,0])
		Xdata = [simulate_ornstein_uhlenbeck(N = N,
									T = T,
									dt = corr_data_generation_params["dt"],
									tau = corr_data_generation_params["tau"]*T,
									mu = corr_data_generation_params["mu"],
									sigma = corr_data_generation_params["sigma"],
									n_dim = corr_data_generation_params["n_dim"],
								    randstate = randstate + i+it_num_1) for i in range(it_num_2)]
		Xdata = np.concatenate(Xdata, axis=2)	
		Xdata = [X, Xdata]
		X = np.concatenate(Xdata, axis=2)

		if it_num_3 != 0:
			random_state = it_num_1+it_num_2 + randstate
			for key in discrete_cov:
				discrete_X = [X,generate_discrete_covariate(discrete_cov[key], N, T, random_state)]
				X = np.concatenate(discrete_X, axis=2)
				random_state += 1
	else:	
		X = stats.norm.rvs(loc = 0.0, 
					       scale = 0.2,
					       size = N*num_covariates,
					       random_state = randstate).reshape(N, num_covariates)
		X = np.repeat(X[:,np.newaxis], T, axis = 1)
	
	#Step2 Generate the survival outcomes
	outcomegen = TimeVaryingSyntheticSurvOutcomes(T = T, p_censored = p_censored)

	fu_times, event_occured = outcomegen(X, beta, randstate, 
										random_baseline_surv = random_baseline_surv, ncc=ncc)

	#Step 3 Sample the longitudinal data at random points
	if sparse:
		num_samp = stats.poisson.rvs(sparse_mean_num_measurements,
								 size = N,
								 random_state = randstate)
		sparse_X = []
		for i in range(N):

			samp_times = stats.randint.rvs(0,fu_times[i]+1,size = num_samp[i], random_state=randstate + i)
			samp_times = np.unique(samp_times)

	
			# redo if we get no samples
			if len(samp_times) == 0:
				j = 1
				while len(samp_times) == 0:
					j += 1
					num_samp_j = stats.poisson.rvs(sparse_mean_num_measurements, size = 1, random_state = randstate + i + j)
					samp_times = stats.randint.rvs(0,fu_times[i]+1,size = num_samp_j, random_state=randstate + i + j)
					samp_times = np.unique(samp_times)
	
	
			sparse_X.append(pd.DataFrame(np.hstack([(np.ones(len(samp_times))*i)[:,np.newaxis],
															 samp_times[:,np.newaxis],
															 X[i][samp_times]])))

		sparse_X = pd.concat(sparse_X).reset_index(drop = True)
		sparse_X.columns = ["id", "time"] + ["X{}".format(i) for i in range(1, num_covariates + 1)]

	else:
		# Or just take the full data
		ids = np.repeat(np.arange(N), T)[:,np.newaxis]
		times = np.hstack(np.repeat(np.arange(T)[:,np.newaxis], N, axis = 1).T)[:,np.newaxis]
		sparse_X = pd.DataFrame(np.hstack([ids, times, X.reshape(T*N, num_covariates)]),
				  		 	    columns = ["id", "time"] + ["X{}".format(i) for i in range(1, num_covariates +1)])

	survdata = pd.DataFrame(np.hstack([fu_times[:,np.newaxis],
									   event_occured[:,np.newaxis]]), 
									   columns = ["time", "event"])
	
	# Add minor jitter
	survdata["time"] = survdata["time"] + stats.norm.rvs(0, (T/1000)**2, size = N, random_state = randstate + 19)
	survdata[survdata["time"] < 0] = 0.0001
	
	return survdata, sparse_X

def check_single_data_gen_params(params: dict):
	key_list = ["dt", "tau", "mu", "sigma", "n_batch"].sort()
	check = list(params.keys()).sort()
	try:
		check == key_list
	except:
		raise ValueError(f"Invalid params dictionary, should contain the following keys: {key_list}") 
	
	return

def check_corr_data_gen_params(params: dict):
	key_list = ["dt", "tau", "mu", "sigma", "n_dim", "n_batch"].sort()
	check = list(params.keys()).sort()
	try:
		check == key_list
	except:
		raise ValueError(f"Invalid params dictionary, should contain the following keys: {key_list}") 
	
	return

def simulate_ornstein_uhlenbeck(N = 1, mu = 0., sigma = 5.0, tau = 0.5, T = 1, dt = 0.001, randstate = 0, n_dim: int = 1):
		"""
		From https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
		Expanded to a vector of N processes with block correlations
		"""
		np.random.seed(randstate)

		n_step = int(T/dt)
		if n_dim == 1:
			sigma_bis= sigma*np.sqrt(2. / tau)
			sqrtdt = np.sqrt(dt)

			x = np.zeros((n_step, N))
			x[0] = stats.norm.rvs(0, 0.5*sigma, size = N, random_state = randstate+1)

			noise = stats.norm.rvs(size = N*n_step, random_state = randstate + 2).reshape(n_step, N)

			for i in range(n_step - 1):
				x[i + 1] = x[i] + dt*(-(x[i] - mu)/tau)
				x[i + 1] += sigma_bis*sqrtdt*noise[i + 1]
			
			return x.T
		
		else:
			if np.sum(tau - tau*np.eye(tau.shape[0])) != 0:
				raise ValueError(f"tau must be a diagonal matrix")

			sigma_bis = sigma @ np.linalg.inv(np.sqrt(tau))*np.sqrt(2)  
			sqrtdt = np.sqrt(dt)

			x = np.zeros((n_step, N, n_dim))
			x[0] = stats.multivariate_normal.rvs(np.zeros(n_dim), sigma, size = N, random_state = randstate+1)

			noise = stats.multivariate_normal.rvs(np.zeros(n_dim), np.eye(n_dim), size = N*n_step, random_state = randstate+2).reshape(n_step, N, n_dim)
			
			for i in range(n_step -1):
					x[i + 1] = x[i] + (dt*(-(x[i] - mu)) @ (np.linalg.inv(tau)))
					x[i + 1] +=  noise[i + 1] @ sigma_bis*sqrtdt

			return x.transpose([1,0,2])


def generate_discrete_covariate(cov_value_prob: list[list], N, T, random_state):
	
	np.random.seed(random_state)
	
	if len(cov_value_prob[0]) < 2:
		raise ValueError(f"discrete variables must have at least 2 values")
	cov_values = np.array(cov_value_prob[0])
	cov_prob = np.array(cov_value_prob[1])
	
	if cov_values.shape[0] == 2:
		cov_values = np.array(cov_value_prob[0])
		cov_prob = np.array(cov_value_prob[1])
		data = np.ones((T,N))
		value_vec = np.random.choice(np.arange(cov_values.shape[0]), size=N, replace=True, p=cov_prob)
		data = data * value_vec
		
		return data[:,:,np.newaxis].transpose([1,0,2])
	
	else:
		cov_values = np.array(cov_value_prob[0])
		cov_prob = np.array(cov_value_prob[1])
		data = np.zeros((T,N,cov_values.shape[0]))
		value_vec = np.random.choice(np.arange(cov_values.shape[0]), size=N, replace=True, p=cov_prob)
		for i in range(value_vec.shape[0]):
			data[:,i,value_vec[i]] = 1
		
		return data.transpose([1,0,2])
	
		
	
	
