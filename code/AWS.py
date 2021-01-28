import numpy as np
from sklearn.ensemble import IsolationForest
import time

def generate_aws_explanations(IF, dataset, verbose=False, 
                              mode="original", check_bag=False, normalize=False):

  """
  Returns a list of AWS explanation vectors.

  Also, returns lists of ordering indices (for plotting) and execution time
  per instance, for the passed dataset.

  Parameters
  ----------
  IF :  Isolation Forest
      The fit Isolation Forest instance 
  dataset : numpy array
      The instances that need to be explained
  verbose : bool, optional
      A flag used to print regular status updates
  mode: str, optional
      (Ignore) A flag used to control the explanation behaviour
  check_bag: bool, optional
      A flag used to check only those Isolation Trees containing the instance 
  normalize: bool, optional
      A flag used to normalize the returned vectors s.t. they add to 1

  """

  #Initialize the result lists
  aws_scores_l = []
  ord_idx_l= []
  exec_time_l = []

  #Iterate over each passed instance 
  for i, data in enumerate(dataset):

    #Generate the explanation and execution time for an instance
    aws_scores, exec_time = point_aws_explanation(IF, data, check_bag, mode)

    #Append the results to the results list
    aws_scores_l.append(aws_scores)
    ord_idx_l.append(np.argsort(aws_scores)[::-1])
    exec_time_l.append(exec_time)
  
  #Normalize results depending on the flag value
  if normalize:
    for i, score in enumerate(aws_scores_l):
      rowsum = np.sum(score)
      aws_scores_l[i] = score/rowsum

  return aws_scores_l, ord_idx_l, exec_time_l

def point_aws_explanation(model, data, check_bag, mode):

  """
  Returns AWS explanation vector for a single instance.

  Parameters
  ----------
  IF :  Isolation Forest
      The fit Isolation Forest instance 
  data : numpy vector
      The specific instance that need to be explained
  check_bag: bool, optional
      A flag used to check only those Isolation Trees containing the instance 
  mode: str, optional
      (Ignore) A flag used to control the explanation behaviour
  """

  #Start tracking execution time
  start = time.time()

  #Initialize the vector that will store the explanation weights
  feature_relevance = np.zeros(shape=data.shape)

  #Iterate over each Isolation Tree of the passed IF model
  for i, estimator in enumerate(model.estimators_):

    #Get the path that the sample has taken
    node_indicator = estimator.decision_path(np.atleast_2d(data))
    node_ids = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    #Get the split features of all nodes that sample passes through
    splits_feature = estimator.tree_.feature[node_ids][:-1]
    
    #Get the size of all nodes that sample passes through
    node_sizes = estimator.tree_.n_node_samples[node_ids]

    #Calculate the relevance of each split as per AWS
    splits_parent = node_sizes[:-1]
    splits_child = node_sizes[1:]
    
    #IGNORE: Deprecated - part of older codebase
    if mode == 'diffi':
      multiplier = (1/len(node_ids) - 1/np.ceil(np.log2(model.max_samples)))
      splits_relevance = np.multiply(np.log2(splits_parent/splits_child), multiplier)
    
    #This part of the conditional is used for AWS
    elif mode == 'clement':
      term = np.log2(splits_parent / splits_child) - 1
      splits_relevance = term
    
    #IGNORE: Deprecated - part of older codebase
    else:
      splits_relevance = np.log2(splits_parent / splits_child)

    #Compute non-normalised feature importance of the tree
    #But first, get the array of unique features & their corresponding indices
    for f in np.nditer(np.unique(splits_feature)):
        feature_relevance[f] += np.mean(splits_relevance[np.where(splits_feature == f)[0]])    

  #Stop tracking execution time
  end = time.time()
  exec_time = end - start
  
  return feature_relevance, exec_time