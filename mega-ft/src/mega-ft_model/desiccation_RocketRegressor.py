#################################################################
## Desiccation with time series models
##
## uses ROCKAT as a regressor (similar to the 'mega-ft' model for defect detection)
#################################################################



# classify in the time series domain
# !pip install aeon or using conda
# https://www.aeon-toolkit.org/en/stable/examples/classification/classification.html


import os
import glob
import scipy.io
from pathlib import Path

from aeon.regression.convolution_based import RocketRegressor
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.regression.feature_based import Catch22Regressor, TSFreshRegressor
from aeon.regression.interval_based import RandomIntervalSpectralEnsembleRegressor, DrCIFRegressor
from sklearn.model_selection import train_test_split

import datetime
import time
import os

import numpy as np

from bricks_analysis_common import *


# select classifiers
N_PARALLEL_JOBS = -1#-1  # -1: use all cores

regressors = [
    ##### distance-based

    # most basic method to do it: 1-NN with Euclidean distance  (data needs to be scaled!!)
#    KNeighborsTimeSeriesRegressor(distance="euclidean", n_neighbors=1),

    ###### time series feature extractors:
#    TSFreshRegressor(default_fc_parameters = "efficient", verbose=True), # other: "comprehensive"

#    Catch22Regressor(catch24=False, n_jobs=N_PARALLEL_JOBS),

#    RandomIntervalSpectralEnsembleRegressor(n_jobs=N_PARALLEL_JOBS),
#    DrCIFRegressor(n_jobs=N_PARALLEL_JOBS), # features from frequency domain

  #  IndividualBOSS(n_jobs=N_PARALLEL_JOBS),

    ### time series ensembles
#   HIVECOTEV2(verbose=True, time_limit_in_minutes=2, n_jobs=N_PARALLEL_JOBS),

    ### ROCKET
    RocketRegressor(num_kernels=10000, n_jobs=N_PARALLEL_JOBS),
    #RocketRegressor(rocket_transform="multirocket", num_kernels=10000, n_jobs=N_PARALLEL_JOBS)
]


#######################################################################################################################
#######################################################################################################################


config.FILES_TO_LOAD_STR_PATTERNS = ["Platform_1_FRONT"]

# specific to this one experiment
desiccation_percentages = [29 ,39 , 51, 63,  74,  80]

# config.CLASS_LABELS_INT_BY_NAME = {str(i)+ "hr" : i for i in range(2,8)}


config.CLASS_LABELS_INT_BY_NAME = {str(i)+ "hr" : desiccation_percentages[i-2] for i in range(2,8)}

config.CLASS_LABELS_NAME_BY_INT = {value: key for key, value in config.CLASS_LABELS_INT_BY_NAME.items()}


str_dateTime = f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
fileName_output = os.path.join(config.RESULTS_PATH, f"bricks_drying_results_response_regression_{str_dateTime}.txt")

dict_files = MatFileLoader(config.DATA_PATH_DESICCATION, files_to_load_str_pattern=config.FILES_TO_LOAD_STR_PATTERNS, fileName_output=fileName_output).load_files()

dataExtractor = DataExtractor(fileName_output)
data, labels, file_names = dataExtractor.transform_loaded_data_to_ML_timeseries_data(dict_files, scale_wrt_hammer=True, class_name_is_prefix = False)

#ml = MachineLearningPipeline(regressors, fileName_output, scale_data=False, balance_data=False)

train_data, test_data, train_labels, test_labels = train_test_split(
                            data, labels, test_size = 0.2, random_state=123)

### train the classifiers and obtain results
for regressor in regressors:
    print(f"Training and regression with: {regressor}")

    regressor.fit(train_data, train_labels)
    y_pred = regressor.predict(test_data)
#    mse = 1/len(test_labels) * np.sum((test_labels - y_pred) * (test_labels - y_pred))

    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

    mse = mean_squared_error(test_labels, y_pred)
    mae = mean_absolute_error(test_labels, y_pred)
    mape = mean_absolute_percentage_error(test_labels, y_pred)

    res = f"\nRegressor: {regressor}\n. MSE: {mse}\n , MAE {mae}\n, MAPE {mape}\n"

    res += "Target variables:\n"
    res += str(test_labels) + "\n"
    res += "Predicted values:\n"
    res += str(y_pred) + "\n"

    print(res)

    with open(fileName_output, "a") as text_file:
        text_file.writelines(f"Regression with {regressor}\n")
        text_file.writelines(res)
