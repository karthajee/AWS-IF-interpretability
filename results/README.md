# Glossary of the Results

This folder contains the .png versions of the experiment results found in our experiment notebook.

We have chosen a naming convention that will help you associate which is which. A typical filename follows a format "[exp_type][num]_[dataset(if applicable)]_[details]:

1. [exp_type] can be:
	1.a. __synexp__ for experiments with synthetic datasets
	1.b. __realexp__ for experiments with real datasets listed in the [DIFFI paper](https://paperswithcode.com/paper/interpretable-anomaly-detection-with-diffi)
	1.c. __heatmap__ for heatmap experiments

2. [num] can take up values from 1..3 to help you associate the plot with the relevant section in the notebook.

3. [dataset(if applicable)] mentions the name of the real-world datasets used for __realexp__ datasets. 

4. [details] help you distinguish different plots connected to the same experiment. E.g. __RMSE__ is used to indicate comparative RMSE plots whereas __time__ indicates plots that compare execution times. When in doubt, do refer to our main notebook to understand what's what!
