#Core Imports for experiments
import shap
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import itertools
from statistics import mean
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from operator import truediv
from collections import defaultdict
import time

#Imports for saving exp files
from datetime import datetime
import json
import pickle

#Imports to parallelize experiment iterations 
from multiprocessing import cpu_count
from multiprocessing import Pool

#DIFFI Imports
import utils
import sklearn_mod_functions
import interpretability_module as interp

#Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore")

#Local DIFFI
def generate_diffi_explanations(model, dataset):
  
  """
  Returns a list of Local DIFFI explanation vectors.

  Also, returns lists of ordering indices (for plotting) and execution time
  per instance, for the passed dataset.

  Parameters
  ----------
  model :  Isolation Forest
      The fit Isolation Forest instance 
  dataset : numpy array
      The instances that need to be explained
  """

  #Produce the explanation vectors as per official DIFFI function
  diffi_l, ord_idx_l, exec_time_l = utils.local_diffi_batch(model, dataset)
  return diffi_l, ord_idx_l, exec_time_l

#SHAP
def generate_shap_explanations(model, dataset):
  
  """
  Returns a list of SHAP explanation vectors.

  Also, returns lists of ordering indices (for plotting) and execution time
  per instance, for the passed dataset.

  Parameters
  ----------
  model :  Isolation Forest
      The fit Isolation Forest instance 
  dataset : numpy array
      The instances that need to be explained
  """

  ##Initialize the result lists
  shap_values_l = []
  ord_idx_l= []
  exec_time_l = []

  #Iterate over each passed instance 
  for i, data in enumerate(dataset):
    
    #Start tracking execution time
    start = time.time()

    #Generate the SHAP explanation for an instance
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(data)
    
    #Stop tracking execution time
    end = time.time()  
    exec_time = end - start

    #Append the results to the results list
    shap_values_l.append(shap_values)
    ord_idx_l.append(np.argsort(shap_values)[::-1])
    exec_time_l.append(exec_time)

  return shap_values_l, ord_idx_l, exec_time_l

"""
Experiment Functions
As per the methodology outlined in the overview, this section houses the relevant functions to:

1. Generate normal datasets.
2. Convert randomly chosen normal instances from the dataset to anomalies.
3. Compute and normalize the difference between expected and actual explanation vectors.
4. Generate random explanations as a baseline
5. Generate ground truth for explanations of selected anomalies
6. Evaluate RMSE loss between actual and expected explanation vectors.
7. Run the experiments.
8. Save the results.
9. Plot the results.
"""

#Generating the normal dataset
def generate_normal_points(points=1000, dimensionality=2, clusters=1, max=20, random_state=42):
  
  """
  Returns a dataset of normal instances.

  Normal instances are grouped into a certain number of clusters.

  Parameters
  ----------
  points : int
      The number of normal instances to be produced
  dimensionality : int
      The number of features of the dataset
  clusters : int
      The number of clusters normal instances are grouped into
  max : int
      A variable that controls range of values from which samples are drawn
  random_state: int
      Setting the random seed for the sampling process
  """

  X, _ = make_blobs(n_samples=points, n_features=dimensionality, 
                             centers=clusters, random_state=random_state, 
                             center_box=(0, max))
  return X

#Anomalising a passed dataset
def anomaliser(dataset, indices, n=1, r_mode='t'):

  """
  Converts the passed normal dataset into an anomalised dataset.

  For the passed instances, certain attributes are randomly but systematically
  picked and their values along these attributes are set to be anomalous.
  
  Parameters
  ----------
  dataset:  numpy array
      The normal dataset
  indices: numpy vector
      The indices of the normal instances, picked to be anomalised
  n : int
      The features of the normal dataset
  r_mode : str
      (Ignore) A flag used to control the behaviour of the anomaliser

  """

  #For each index, randomly pick a feature OR a combination of n features
  N = range(0, n)
  if r_mode == 't':
    setting_l = random.choices(N, k=len(indices))
  else:
    setting_l = [n-1] * len(indices)
  
  
  #Generate a list of list of attributes, where each inner list contains
  #the attributes along which the instances will be anomalised across
  picked_l = [random.sample(range(dataset.shape[1]), setting+1) for setting in setting_l]

  #Make the instances anomalous across those feature(s). We use noise in the 
  #multiplicand to ensure that the anomalous points do not have same values
  ano_rand_points = np.array([[np.random.normal(3, 0.3) * np.amax(dataset[:,el]) if el in picked 
                            else dataset[index, el] for el in range(dataset.shape[1])]
                            for (index, picked) in zip(indices, picked_l)])

  #Create a copy of the dataset and save the points
  new_dataset=np.copy(dataset)
  new_dataset[indices]=ano_rand_points
  
  return new_dataset, setting_l, picked_l

#Generating the normalised difference vector
def generate_normed_diff(m1, m2):

  """
  Returns the normalised row-wise difference of 2 matrix.

  Parameters
  ----------
  m1 : numpy matrix
      The first matrix
  m2 : numpy matrix
      The second matrix, which is to be subtracted from the first matrix
  """

  #Calculate the difference between the 2 matrices
  diff = m2 - m1

  #We make negative elements in the difference matrix to be positive.
  #This is to make normalization possible as otherwise, all positive
  #elements will be set to 1 and all negative elements will be set to 0.
  pos_diff = np.abs(diff)
  
  #Normalize the difference matrix
  row_sum = np.sum(pos_diff, axis=1)
  norm_diff = pos_diff / row_sum[:, np.newaxis]

  return diff, pos_diff, norm_diff

#Model a random explainer
def random_explainer(rand_indices, n_features):

  """
  Returns a matrix of row-normalized random vectors.

  Parameters
  ----------
  rand_indices: numpy vector
      The indices of the normal instances, picked to be anomalised
  n_features: int
      The number of features of the instances
  """
  
  #Sample from a uniform distribution to get values between 0 and 1
  uniform = np.random.uniform(0, 1, size=(len(rand_indices), n_features))

  #Normalise them to get a probability distribution
  random_explanations = uniform/np.sum(uniform, axis=1)[:, np.newaxis]
  
  return random_explanations

#Generate ground truth
def generate_ground_truth(rand_indices, picked_feature_l, n_features):
  
  """
  Generate ground truth from the list of picked features.

  If only 1 feature is picked, that receives a value of 1, rest 0
  If 2 features are picked, both receive a value of 0.5, rest 0. So on.

  Parameters
  ----------
  rand_indices : numpy vector
      The indices of the normal instances, picked to be anomalised
  picked_feature_l: list
      The list of picked (list of) attributes for each picked normal instance
  n_features: int
      The number of features of the instances
  """
  
  ground_truth = np.zeros(shape=(len(rand_indices), n_features))
  
  for i, picked in enumerate(picked_feature_l):
      ground_truth[i, picked] = 1/len(picked)
  
  return ground_truth

#Evaluate loss
def evaluate_loss(model_outputs, ground_truth, n_ano_points):
  
  """
  Return the RMSE loss between Isolation Forest outputs and ground truth.

  Parameters
  ----------
  model_outputs: numpy matrix
      The anomaly scores for those anomalous instances by the fit IF model
  ground_truth: numpy matrix
      The generated ground truth
  n_ano_points: int
      The number of normal instances, picked to be anomalised
  """
  #Initialize the result list
  rmse_l = []
  
  #Iterate over each instance and compare with ground truth
  for (output, label) in zip(model_outputs, ground_truth):
    
    #Calculate RMSE loss for the instance
    rmse = np.linalg.norm(label - output)
    rmse_l.append(rmse)

  #Append the results to the results list
  total_rmse_loss = np.sum(rmse_l)/np.sqrt(n_ano_points)

  return total_rmse_loss

#Main experiment function per iteration
def run_experiment_single(iter_count, exp_string, n_points=None, n_features=None, n=None,
                          data=None, n_est=100, contam_rate=0.1, r_mode='t', max_samples=None):
  
  """
  Runs an experiment for a single instance.

  We are interested in running experiments that study the influence of 
  increasing dataset size, dimensionality and number of features. Additionally, 
  we also want to study how the AWS method fares to SHAP and DIFFI fordifferent feature subsets in real
  datasets mentioned in the DIFFI paper.

  Parameters
  ----------
  iter_count : int
      The iteration count for printing purposes.
  exp_string : str
      The codified flag that communicates the experiment we are executing.
      Codebook:
      syn1 - Experiment to study the effect of increasing dataset size
      syn2 - Experiment to study the effect of increasing dimensionality
      syn3 - Experiment to study the effect of increasing anomalised features
      real1 & real2 - Experiment to study real-world datasets
  n_points : int, bool
      The size of the real world dataset. None for synthetic.
  n_features : int, bool
      The number of features of the real world dataset. None for synthetic.
  n : int, bool
      The number of features that can be picked to anomalise. None for synthetic.
  data : Pandas dataframe
      The real world dataset. None for synthetic.
  n_est : int
      The number of Isolation Trees in the Forest
  contam_rate: float
      The proportion of instances that are anomalous
  r_mode : str
      (Ignore) A flag used to control the behaviour of the anomaliser
  max_samples: int
      The bootstrap sample size for each Isolation Tree of the Forest
  """

  #Initialize the result dictionary
  single_result_dict = {'index': iter_count}
  
  #Use iter_count to initialize subprocess RNG instance
  rng = np.random.RandomState(iter_count)

  #Print status updates 
  if exp_string == 'syn1':
      print(f"[INFO] Iteration {iter_count+1}: Dataset Size - {n_points}...")
  elif exp_string == 'syn2':
      print(f"[INFO] Iteration {iter_count+1}: Dataset Dimensionality - {n_features} features...")
  elif exp_string == 'syn3':
      print(f"[INFO] Iteration {iter_count+1}: Upto {n} features anomalised for dataset of {n_points:,} samples with {n_features} features...")
  elif exp_string[:5] == 'real1':
      print(f"[INFO] Iteration {iter_count+1}: {exp_string[6:]} dataset ({data.shape[0]} samples and {data.shape[1]} features)...")
  else:
      print(f"[INFO] Iteration {iter_count+1}: Upto {n} features anomalised for {exp_string[6:]} dataset ({data.shape[0]} samples and {data.shape[1]} features)...")

  #Generate synthetic dataset for synthetic experiments
  if isinstance(data, type(None)):
    n_clusters= n_features // 2
    max_box = 20 * (n_clusters//4 + 1)
    data = generate_normal_points(points=n_points, dimensionality=n_features, 
                                          clusters=n_clusters, max=max_box, random_state=rng)
    single_result_dict['orig_dataset'] = data
    max_samples = min(256, data.shape[0])
    n = data.shape[1]
    data = pd.DataFrame(data)
  else:
    single_result_dict['orig_dataset'] = []

  #Pick n_ano_points instances at random from the dataset
  n_ano_points = int(np.ceil(contam_rate * data.shape[0]))
  rand_indices = rng.randint(low=0, high=int(data.shape[0]), size=n_ano_points)
  single_result_dict['rand_indices'] = rand_indices
  
  #Fit the model to the dataset
  clf_orig = IsolationForest(n_estimators=n_est, max_samples=max_samples, random_state=rng)
  clf_orig.fit(data.values)

  #Generate original explanation matrices
  orig_aws_l, _, orig_aws_exec_time_l = generate_aws_explanations(clf_orig, data.values[rand_indices])
  orig_aws_clem_l, _, orig_aws_clem_exec_time_l = generate_aws_explanations(clf_orig, data.values[rand_indices], mode='clement')
  orig_aws_dif_l, _, orig_aws_dif_exec_time_l = generate_aws_explanations(clf_orig, data.values[rand_indices], mode='diffi')
  orig_shap_l, _, orig_shap_exec_time_l = generate_shap_explanations(clf_orig, data.values[rand_indices])
  orig_diffi_l, _, orig_diffi_exec_time_l = generate_diffi_explanations(clf_orig, data.values[rand_indices])

  #Convert the list to numpy array 
  orig_aws_exp_matrix = np.array(orig_aws_l)
  orig_aws_clem_exp_matrix = np.array(orig_aws_clem_l)
  orig_aws_dif_exp_matrix = np.array(orig_aws_dif_l)
  orig_shap_exp_matrix = np.array(orig_shap_l)
  orig_diffi_exp_matrix = np.array(orig_diffi_l)

  single_result_dict['orig_aws_exp_matrix'] = orig_aws_exp_matrix
  single_result_dict['orig_aws_clem_exp_matrix'] = orig_aws_clem_exp_matrix
  single_result_dict['orig_aws_dif_exp_matrix'] = orig_aws_dif_exp_matrix
  single_result_dict['orig_shap_exp_matrix'] = orig_shap_exp_matrix
  single_result_dict['orig_diffi_exp_matrix'] = orig_diffi_exp_matrix

  #Anomalise the dataset
  new_dataset, settings_l, features_l = anomaliser(data.values, rand_indices, n, r_mode)
  single_result_dict['new_dataset']= new_dataset
  single_result_dict['settings_l']= settings_l
  single_result_dict['features_l']= features_l

  #Fit the model to the dataset
  clf_new = IsolationForest(n_estimators=n_est, max_samples=max_samples, contamination=contam_rate, random_state=rng)
  clf_new.fit(new_dataset)

  #Generate new explanation matrices
  new_aws_l, _, new_aws_exec_time_l = generate_aws_explanations(clf_new, new_dataset[rand_indices])
  new_aws_clem_l, _, new_aws_clem_exec_time_l = generate_aws_explanations(clf_new, new_dataset[rand_indices], mode='clement')
  new_aws_dif_l, _, new_aws_dif_exec_time_l = generate_aws_explanations(clf_new, new_dataset[rand_indices], mode='diffi')
  new_shap_l, _, new_shap_exec_time_l = generate_shap_explanations(clf_new, new_dataset[rand_indices])
  new_diffi_l, _, new_diffi_exec_time_l = generate_diffi_explanations(clf_new, new_dataset[rand_indices])

  new_aws_exp_matrix = np.array(new_aws_l)
  new_aws_clem_exp_matrix = np.array(new_aws_clem_l)
  new_aws_dif_exp_matrix = np.array(new_aws_dif_l)
  new_shap_exp_matrix = np.array(new_shap_l)
  new_diffi_exp_matrix = np.array(new_diffi_l)

  single_result_dict['new_aws_exp_matrix'] = new_aws_exp_matrix
  single_result_dict['new_aws_clem_exp_matrix'] = new_aws_clem_exp_matrix
  single_result_dict['new_aws_dif_exp_matrix'] = new_aws_dif_exp_matrix
  single_result_dict['new_shap_exp_matrix'] = new_shap_exp_matrix
  single_result_dict['new_diffi_exp_matrix'] = new_diffi_exp_matrix

  #Get the difference of the 2 numpy matrices
  _, _, aws_norm_diff = generate_normed_diff(m1=orig_aws_exp_matrix, 
                                                    m2=new_aws_exp_matrix)
  _, _, aws_clem_norm_diff = generate_normed_diff(m1=orig_aws_clem_exp_matrix, 
                                                  m2=new_aws_clem_exp_matrix)
  _, _, aws_dif_norm_diff = generate_normed_diff(m1=orig_aws_dif_exp_matrix, 
                                                  m2=new_aws_dif_exp_matrix)
  _, _, shap_norm_diff = generate_normed_diff(m1=orig_shap_exp_matrix, 
                                                    m2=new_shap_exp_matrix)
  _, _, diffi_norm_diff = generate_normed_diff(m1=orig_diffi_exp_matrix, 
                                                    m2=new_diffi_exp_matrix)
  random_exp_matrix = random_explainer(rand_indices, data.shape[1])

  single_result_dict['aws_norm_diff'] = aws_norm_diff
  single_result_dict['aws_clem_norm_diff'] = aws_clem_norm_diff
  single_result_dict['aws_dif_norm_diff'] = aws_dif_norm_diff
  single_result_dict['shap_norm_diff'] = shap_norm_diff
  single_result_dict['diffi_norm_diff'] = diffi_norm_diff
  single_result_dict['random_exp_matrix'] = random_exp_matrix

  #Generate ground truth
  ground_truth = generate_ground_truth(rand_indices, features_l, data.shape[1])
  single_result_dict['ground_truth'] = ground_truth

  #Compute the RMSE loss
  aws_loss = evaluate_loss(aws_norm_diff, ground_truth, n_ano_points)
  aws_clem_loss = evaluate_loss(aws_clem_norm_diff, ground_truth, n_ano_points)
  aws_dif_loss = evaluate_loss(aws_dif_norm_diff, ground_truth, n_ano_points)
  shap_loss = evaluate_loss(shap_norm_diff, ground_truth, n_ano_points)
  diffi_loss = evaluate_loss(diffi_norm_diff, ground_truth, n_ano_points)
  random_loss = evaluate_loss(random_exp_matrix, ground_truth, n_ano_points)

  single_result_dict['aws_rmse_loss'] = aws_loss
  single_result_dict['aws_clem_rmse_loss'] = aws_clem_loss
  single_result_dict['aws_dif_rmse_loss'] = aws_dif_loss
  single_result_dict['shap_rmse_loss'] = shap_loss
  single_result_dict['diffi_rmse_loss'] = diffi_loss
  single_result_dict['random_rmse_loss'] = random_loss
  
  #Compute the Execution Time
  aws_exec_time = mean(orig_aws_exec_time_l + new_aws_exec_time_l)
  aws_clem_exec_time = mean(orig_aws_clem_exec_time_l + new_aws_clem_exec_time_l)
  aws_dif_exec_time = mean(orig_aws_dif_exec_time_l + new_aws_dif_exec_time_l)
  shap_exec_time = mean(orig_shap_exec_time_l + new_shap_exec_time_l)
  diffi_exec_time = mean(orig_diffi_exec_time_l + new_diffi_exec_time_l)

  single_result_dict['aws_exec_time'] = aws_exec_time
  single_result_dict['aws_clem_exec_time'] = aws_clem_exec_time
  single_result_dict['aws_dif_exec_time'] = aws_dif_exec_time
  single_result_dict['shap_exec_time'] = shap_exec_time
  single_result_dict['diffi_exec_time'] = diffi_exec_time

  return single_result_dict

#Batch experiment runs
def run_experiment_batch(exp_string, data=None, n_est=100, contam_rate=0.1,
                         max_points=5000, point_step=500, max_features=25,
                         feature_step=2, max_exps=20, r_mode='t', normalize=False, max_samples=None):
  
  """
  Runs an experiment for a batch of instances.

  We are interested in running experiments that study the influence of 
  increasing dataset size, dimensionality and number of features. Additionally, 
  we also want to study how the AWS method fares to SHAP and DIFFI fordifferent feature subsets in real
  datasets mentioned in the DIFFI paper.

  Parameters
  ----------
  exp_string : str
      The codified flag that communicates the experiment we are executing.
      Codebook:
      syn1 - Experiment to study the effect of increasing dataset size
      syn2 - Experiment to study the effect of increasing dimensionality
      syn3 - Experiment to study the effect of increasing anomalised features
      real1 & real2 - Experiment to study real-world datasets
  data : Pandas dataframe
      The real world dataset. None for synthetic.
  n_est : int
      The number of Isolation Trees in the Forest
  contam_rate: float
      The proportion of instances that are anomalous
  max_points : int, bool
      The max dataset to be considered for 'syn1' exp. None for others.
  point_step: int, bool
      The step size for 'syn1' experiment
  max_features : int, bool
      The number of features of the real world dataset. None for synthetic.
  feature_step: int, bool
      The step size for 'syn1' experiment
  max_exps : int
      (Ignore) Indicate total number of experiments to be conducted
  r_mode : str
      (Ignore) A flag used to control the behaviour of the anomaliser
  normalize : bool
      A flag used to normalize returned AWS vectors s.t. they add to 1
  max_samples: int
      The bootstrap sample size for each Isolation Tree of the Forest 
  """

  #Set random seed for random. Note numpy random seed behaves differently
  #Each child inherits the same random state as parent when forking
  random.seed(seed_val)
  
  #To Store Main Experiment Results
  random_loss_l = []
  aws_loss_l = []
  aws_clem_loss_l = []
  aws_dif_loss_l = []
  diffi_loss_l = []
  shap_loss_l = []

  aws_exec_time_l = []
  aws_clem_exec_time_l = []
  aws_dif_exec_time_l = []
  diffi_exec_time_l = []
  shap_exec_time_l = []

  #To Store Other Experiment Results
  orig_dataset_l = []
  new_dataset_l = []
  rand_indices_l = []
  settings_ll = []
  features_ll = []
  ground_truth_l = []

  orig_aws_exp_matrix_l = []
  orig_aws_clem_exp_matrix_l = []
  orig_aws_dif_exp_matrix_l = []
  orig_shap_exp_matrix_l = []
  orig_diffi_exp_matrix_l = []

  new_aws_exp_matrix_l = []
  new_aws_clem_exp_matrix_l = []
  new_aws_dif_exp_matrix_l = []
  new_shap_exp_matrix_l = []
  new_diffi_exp_matrix_l = []

  aws_norm_diff_l = []
  aws_clem_norm_diff_l = []
  aws_dif_norm_diff_l = []
  diffi_norm_diff_l = []
  shap_norm_diff_l = []
  random_norm_diff_l = []
  
  #For saving purposes - 
  save_string = exp_string + "_" + str(max_features) + "features_" + str(max_points) + "maxPoints_rmode=" + r_mode
  
  #Initialize the multiprocessing pool iterable that needs to be iterated over
  #The itertable is dependent on the type of experiment we are executing
  if exp_string == 'syn1':
    iterable = range(1000, max_points, point_step)
    pool_iterable = [(i, exp_string, val, max_features, None, None, n_est, contam_rate, r_mode, max_samples) 
                    for i, val in enumerate(iterable)]
    label = 'Dataset Size'

  elif exp_string == 'syn2':
    iterable = range(2, max_features, feature_step)
    pool_iterable = [(i, exp_string, max_points, val, None, None, n_est, contam_rate, r_mode, max_samples) 
                    for i, val in enumerate(iterable)]
    label = 'Number Of Attributes In The Dataset'

  elif exp_string == 'syn3':
    iterable = range(1, max_features)
    pool_iterable = [(i, exp_string, max_points, max_features, val, None, n_est, contam_rate, r_mode, max_samples)
                     for i, val in enumerate(iterable)]
    #label = 'Max Number Of Candidate Attributes For Anomalisation' if r_mode == 't' else 'Number Of Anomalised Attributes'
    label = 'Number Of Anomalised Attributes'
    
  elif exp_string[:5] == 'real1':
    iterable = range(0, max_exps)
    pool_iterable = [(i, exp_string, None, None, None, data, n_est, contam_rate, r_mode, max_samples)
                     for i in iterable]
    save_string = exp_string + "_" + r_mode #Change the string only if real dataset
    label = 'Iteration Count'
    normalize = True

  elif exp_string[:5] == 'real2':
    iterable = range(1, data.shape[1])
    pool_iterable = [(i, exp_string, None, None, val, data, n_est, contam_rate, r_mode, max_samples)
                     for i, val in enumerate(iterable)]
    
    save_string = exp_string + "_" + r_mode #Change the string only if real
    #label = 'Max Number Of Candidate Attributes For Anomalisation' if r_mode == 't' else 'Number Of Anomalised Attributes'
    label = 'Number Of Anomalised Attributes'
    normalize = True

  else:
    raise ValueError("Invalid experiment details passed!")
  
  #Use the multiprocessing library to parallelize and run experiments
  with Pool(cpu_count()) as p:
      result_map = p.starmap(run_experiment_single, pool_iterable)

  #Convert the map object FIRST into a list of dictionaries, and then 
  #the list of dictionaries into a dictionary of lists
  result = defaultdict(list)
  result_list = list(result_map)
  result_list_sorted = sorted(result_list, key=lambda d: d['index'])
  {result[key].append(single_iter_dict[key]) for single_iter_dict in result_list_sorted for key in single_iter_dict.keys()}

  #For saving
  orig_dataset_l = result['orig_dataset']
  rand_indices_l = result['rand_indices']
  orig_aws_exp_matrix_l = result['orig_aws_exp_matrix']
  orig_aws_clem_exp_matrix_l = result['orig_aws_clem_exp_matrix']
  orig_aws_dif_exp_matrix_l = result['orig_aws_dif_exp_matrix']
  orig_shap_exp_matrix_l = result['orig_shap_exp_matrix']
  orig_diffi_exp_matrix_l = result['orig_diffi_exp_matrix']

  new_dataset_l = result['new_dataset']
  settings_ll = result['settings_l']
  features_ll = result['features_l']

  new_aws_exp_matrix_l = result['new_aws_exp_matrix']
  new_aws_clem_exp_matrix_l = result['new_aws_clem_exp_matrix']
  new_aws_dif_exp_matrix_l = result['new_aws_dif_exp_matrix']
  new_shap_exp_matrix_l = result['new_shap_exp_matrix']
  new_diffi_exp_matrix_l = result['new_diffi_exp_matrix']

  aws_norm_diff_l = result['aws_norm_diff']
  aws_clem_norm_diff_l = result['aws_clem_norm_diff']
  aws_dif_norm_diff_l = result['aws_dif_norm_diff']
  shap_norm_diff_l = result['shap_norm_diff']
  diffi_norm_diff_l = result['diffi_norm_diff']
  random_norm_diff_l = result['random_exp_matrix']

  ground_truth_l = result['ground_truth']

  random_loss_l = result['random_rmse_loss']
  aws_loss_l = result['aws_rmse_loss']
  aws_clem_loss_l = result['aws_clem_rmse_loss']
  aws_dif_loss_l = result['aws_dif_rmse_loss']
  diffi_loss_l = result['diffi_rmse_loss']
  shap_loss_l = result['shap_rmse_loss']

  aws_exec_time_l = result['aws_exec_time']
  aws_clem_exec_time_l = result['aws_clem_exec_time']
  aws_dif_exec_time_l = result['aws_dif_exec_time']
  shap_exec_time_l = result['shap_exec_time']
  diffi_exec_time_l = result['diffi_exec_time']

  total_random_rmse_loss = np.sum(random_loss_l)
  total_aws_rmse_loss = np.sum(aws_loss_l)
  total_aws_clem_rmse_loss = np.sum(aws_clem_loss_l)
  total_aws_dif_rmse_loss = np.sum(aws_dif_loss_l) 
  total_diffi_rmse_loss = np.sum(diffi_loss_l)
  total_shap_rmse_loss = np.sum(shap_loss_l)

  avg_aws_exec_time = mean(aws_exec_time_l)
  avg_aws_clem_exec_time = mean(aws_clem_exec_time_l)
  avg_aws_dif_exec_time = mean(aws_dif_exec_time_l)
  avg_shap_exec_time = mean(shap_exec_time_l)
  avg_diffi_exec_time = mean(diffi_exec_time_l)

  #Printing final status update
  print("\n[INFO] Final Loss Analysis...")
  print(f"Our method [AWS] resulted in an RMSE loss of {total_aws_clem_rmse_loss}")
  print(f"SHAP method resulted in an RMSE loss of {total_shap_rmse_loss}")
  print(f"DIFFI method resulted in an RMSE loss of {total_diffi_rmse_loss}")
  print(f"Random explainer resulted in an RMSE loss of {total_random_rmse_loss}")

  print("\n[INFO] Final Time Analysis...")
  print(f"Our method [AWS]: {avg_aws_clem_exec_time}")
  print(f"SHAP method: {avg_shap_exec_time}")
  print(f"DIFFI method: {avg_diffi_exec_time}\n")

  #Saving the files
  list_of_main_files = ['random_loss_l','aws_loss_l','aws_clem_loss_l','aws_dif_loss_l','diffi_loss_l',
                           'shap_loss_l','aws_exec_time_l','aws_clem_exec_time_l','aws_dif_exec_time_l',
                           'diffi_exec_time_l','shap_exec_time_l']
  
  list_of_support_files = ['orig_dataset_l','new_dataset_l', 'rand_indices_l','settings_ll','features_ll',
                           'ground_truth_l','orig_aws_exp_matrix_l','orig_aws_clem_exp_matrix_l','orig_aws_dif_exp_matrix_l',
                           'orig_shap_exp_matrix_l','orig_diffi_exp_matrix_l','new_aws_exp_matrix_l','new_aws_clem_exp_matrix_l',
                           'new_aws_dif_exp_matrix_l','new_shap_exp_matrix_l','new_diffi_exp_matrix_l','aws_norm_diff_l',
                           'aws_clem_norm_diff_l','aws_dif_norm_diff_l','diffi_norm_diff_l','shap_norm_diff_l', 'random_norm_diff_l']

  (batch_results_dict_main, batch_results_dict_support) = save_files(list_of_main_files, list_of_support_files, save_string, locals())
  
  #Plotting the results
  result_plotter(lists_of_files=[[aws_clem_loss_l, diffi_loss_l, shap_loss_l, random_loss_l],
                [aws_clem_exec_time_l, shap_exec_time_l, diffi_exec_time_l]], 
                plot_mode=['loss', 'time'], 
                fig_save_names=[exp_string + '_Comparing Loss Across Methods', exp_string + '_Comparing Execution Time Across Methods'],
                plot_titles = ['Comparing Loss Across Methods','Comparing Execution Time Across Methods'],
                x_label=label, x_range=iterable, normalize=normalize)

  return batch_results_dict_main, batch_results_dict_support

#Saving the Files
def save_files(main_list, support_list, save_string, val):
  
  """
  Saves the experiment results as json files for later use.

  Parameters
  ----------
  main_list : list
      The list of variable filenames corresponding to primary experiment results
  support_list : list
      The list of variable filenames corresponding to secondary experiment results
  save_string : str
      A string used for setting the name of the result json files
  val : dict
      A dictionary that contains local variables. Each key corresponds to
      a variable name and the corresponding value, the value of the variable
  """
  #Utility function to make modifications to variable values
  #that cannot be DIRECTLY saved in a json file (e.g. numpy arrays ...
  #... need to be converted into lists for json saving to be possible!)
  def selector(x):
    if isinstance(x, np.ndarray):
      return x.tolist()
    elif isinstance(x, np.float64):
      return x.item()
    elif isinstance(x, list):
      if not x:
        return "Real Dataset"
      if isinstance(x[0], np.ndarray):
        return [l.tolist() for l in x]
      else:
        return x
    else:
        return x

  #Extract the dictionaries that will contain the information for
  #main and secondary experiment results
  results_main = {filename: selector(val[filename]) for filename in main_list}
  results_support = {filename: selector(val[filename]) for filename in support_list}
  
  #Use the current time to make dynamic alterations to a standard string
  now = datetime.now()
  date_time = now.strftime("%d-%m_%H:%M")
  save_string += "_" + date_time

  #Change cwd to the folder created in the beginning of the session
  #in case it switched to elsewhere
  if savefiles:
    os.chdir(dir_path)
    
    results_main_name = 'result_main_' + save_string + ".json"
    results_support_name = 'result_support_' + save_string + ".json"
    
    for (content, name) in ((results_main, results_main_name), (results_support, results_support_name)):
      with open(name, 'w') as out_file:
        json.dump(content, out_file, indent=4)

  return (results_main, results_support)

#Plotting the Results
def result_plotter(lists_of_files, plot_mode, plot_titles, fig_save_names, x_label, x_range=None, normalize=False):

  """
  Generate RMSE and Execution Time plots for different experiments.

  Parameters
  ----------
  lists_of_files : list
      A 3-level list that contains the list of loss and execution time results
      for different explanation methods
  plot_mode : list of str
      A flag that is used to determine whether we are plotting RMSE loss or 
      Execution Time
  plot_titles : list of str
      The list of titles for each plot
  fig_save_names : list of str
      The list of filenames for saving purposes
  x_label : str
      X-axis label
  x_range : range object
      The range of X-values for the plot
  normalize : bool
      A flag to determine whether we are plotting absolute RMSE or RMSE 
      standardized to the output of the random explainer. 
  """

  #Set date & time to be appended to the saved file names
  now = datetime.now()
  date_time = now.strftime("%d-%m_%H:%M")

  #Set style for easy and good plot readability
  plt.style.use('seaborn')
  sns.set_context('paper', font_scale=1.4, rc={"lines.linewidth": 1.4})
  
  #Iterate over the lists of files
  for i, list_of_files in enumerate(lists_of_files):
    
    #Check whether we are plotting RMSE loss
    if plot_mode[i] =='loss':
      
      (aws_clem_loss_l, diffi_loss_l, shap_loss_l, random_loss_l) = list_of_files
      fig1, axs1 = plt.subplots(figsize=(10, 5))

      #If normalize is set to True, we standardize RMSE loss of different
      #methods on the basis of the output of the random explainer
      if normalize:
        aws_clem_loss_l = list(map(truediv, aws_clem_loss_l, random_loss_l))
        diffi_loss_l = list(map(truediv, diffi_loss_l, random_loss_l))
        shap_loss_l = list(map(truediv, shap_loss_l, random_loss_l))
        random_loss_l = list(map(truediv, random_loss_l, random_loss_l))
        axs1.set(ylabel='Normalized RMSE', xlabel=x_label)      
      else:
        axs1.set(ylabel='RMSE', xlabel=x_label)

      #Generate the plots
      axs1.plot(x_range, aws_clem_loss_l, label="Our Method: AWS", marker='o')
      axs1.plot(x_range, diffi_loss_l, label="DIFFI", marker='o')
      axs1.plot(x_range, shap_loss_l, label="SHAP", marker='o')
      axs1.plot(x_range, random_loss_l, label="Random Explanation", marker='o')
      
      axs1.set_aspect(1.0/axs1.get_data_ratio(), adjustable='box')
      axs1.legend(title="Methods", fancybox=True, loc='best')
      fig1.tight_layout()  
      
      if savefiles:
        fig1.savefig(fig_save_names[i] + "_RMSE_" + date_time)
    
    #Check whether we are plotting execution time
    elif plot_mode[i] == 'time':

      (aws_clem_exec_time_l, shap_exec_time_l, diffi_exec_time_l) = list_of_files
      fig, axs = plt.subplots(figsize=(10, 5))
      
      #Generate the plots
      axs.plot(x_range, aws_clem_exec_time_l, label="Our Method: AWS", marker='o')
      axs.plot(x_range, diffi_exec_time_l, label="DIFFI", marker='o')
      axs.plot(x_range, shap_exec_time_l, label="SHAP", marker='o')
      
      axs.set_aspect(1.0/axs.get_data_ratio(), adjustable='box')
      axs.set(ylabel='Execution Time Per Sample (s)', xlabel=x_label)
      axs.legend(title="Methods", fancybox=True, loc='best')
      fig.tight_layout()
      if savefiles:
        fig.savefig(fig_save_names[i] + "_" + date_time)
    
    else:
      print(f"Invalid {i}th Argument sent to function call...")
      
  return