#################################################################
## Defect detection with time series models
##
## uses various classifiers from the aeon-toolkit
## final model 'mega-ft' uses ROCKET classifier from aeon-toolkit
#################################################################

import datetime
import time
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bricks_analysis_common import *
import random

random.seed(42)  # set random seed for reproducible results!



def _shuffle_and_split(dat2split, test_size):
    index_of_indices = np.arange(len(dat2split))

    np.random.default_rng().shuffle(index_of_indices)
    dat = np.array(dat2split)[index_of_indices]
    split_index = int(len(dat) * (1 - test_size))
    train = dat[:split_index]
    test = dat[split_index:]
    return train, test

def separate_normal_and_defects_file_names(file_names):
    fn_normal_Boolean = np.array([file_name.startswith("normal") for file_name in file_names])
    idx_normal = np.where(fn_normal_Boolean == True)  # get indices
    idx_defect = np.where(fn_normal_Boolean == False)

    fn_normal = np.array(file_names)[idx_normal]
    fn_defect = np.array(file_names)[idx_defect]

    fn_train_normal, fn_test_normal = _shuffle_and_split(fn_normal, 0.5)
    fn_train_defect, fn_test_defect = _shuffle_and_split(fn_defect, 0.5)
    assert( (len(fn_train_normal) + len(fn_test_normal) + len(fn_train_defect) + len(fn_test_defect)) == len(file_names) )

    return fn_train_normal, fn_test_normal, fn_train_defect, fn_test_defect


def plot_results(df):
    """ uses DataFrame with fixed columns! """
    df_sorted = df.sort_values(by="prediction")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(df_sorted)
    # print(df)

    plt.scatter(df["class label"], df["prediction"], marker='o', color='b')
    plt.xlabel("class label")
    plt.ylabel("prediction")
    plt.title("Anomaly scores w.r.t. classes")
    plt.show()

    import seaborn as sns
    sns.kdeplot(data=df, x="prediction", hue="class label", fill=True)
    plt.title("prediction w.r.t. classes")
    plt.show()

    sns.boxplot(x="class label", y="prediction", data=df)
    plt.title("prediction w.r.t. classes")
    plt.show()

    plt.plot(df["prediction"], marker='o')
    plt.ylabel("prediction")
    plt.title("predictions")
    plt.show()

################## combine both fabrications -- classification: 2-class (normal vs. defects); classification: all 9-classes
print("#####################################################")


params_scaling = [False, True]
params_classes = ["2-class", "9-class"]

for scale_wrt_hammer in params_scaling:
    for par_classes in params_classes:

        str_dateTime = f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
        fileName_output = os.path.join(config.RESULTS_PATH, f"bricks_results_response_{par_classes}_s1_h3_scaling-wrt-hammer-is-{scale_wrt_hammer}_no-balanced-data_80_20_testsplit_{str_dateTime}.txt")

        # use data from fabrication 1 and 2 + additonally data from fabrication 3 and 4 (normal bricks only)
        fileName_output = os.path.join(config.RESULTS_PATH, f"bricks_results_response_jrnl_{par_classes}_s1_h3_scaling-wrt-hammer-is-{scale_wrt_hammer}_no-balanced-data_80_20_testsplit_{str_dateTime}.txt")
        dict_files = MatFileLoader(config.DATA_PATH_DEFECT_DETECTION, files_to_load_str_pattern=config.FILES_TO_LOAD_STR_PATTERNS, fileName_output=fileName_output).load_files()

        dataExtractor = DataExtractor(fileName_output)
        data, labels, file_names = dataExtractor.transform_loaded_data_to_ML_timeseries_data(dict_files, scale_wrt_hammer=scale_wrt_hammer)

        ml = MachineLearningPipeline(classifiers, fileName_output, scale_data=False, balance_data=False)

        if par_classes == "2-class":
            # to be compatible with sklearn outlier detectors (sklearn uses +1 for normal and -1 for anomaly/outlier)
            labels = np.array([1 if lab == 0 else -1 for lab in labels])

        train_data, test_data, train_labels, test_labels = train_test_split(
                                        data, labels, test_size=0.2, random_state=123)

        res_by_classif = ml.train_and_classify(train_data, test_data, train_labels, test_labels,
                              test_file_names=None, description="train: all classes fab1+fab2, test: all classes fab1+fab2")   #=> works

        #plot_results(res_by_classif[str(classifiers[0]))


