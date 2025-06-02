#################################################################
## Anomaly detection with time series model
##
## uses ROCKAD (anomaly detection variant of the ROCKET classifier)
#################################################################

import os
import glob
import random

import scipy.io
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

### one-class classification, i.e. whole series anomaly detection
# training on data from the normal class only

from ROCKAD import ROCKAD   # uses own implementation of ROCKAD

import os
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from bricks_analysis_common import *


###############
def _shuffle_and_split(dat2split, test_size):
    index_of_indices = np.arange(len(dat2split))

    np.random.default_rng().shuffle(index_of_indices)
    dat = np.array(dat2split)[index_of_indices]
    split_index = int(len(dat) * (1 - test_size))
    train = dat[:split_index]
    test = dat[split_index:]
    return train, test

one_class_classifiers = [
#ROCKAD(rocket_transformer_name="multirocket") # with default params
ROCKAD() # variant used: with default params
#ROCKAD(n_kernels = 10000, n_neighbors = 5, power_transform = True, random_state = 42)
]

###########
split = "random"
#split = "split-by-specimen"

str_dateTime = f"{datetime.datetime.now():%Y-%m-%d_%H-%M}"
fileName_output = os.path.join(config.RESULTS_PATH, f"bricks_results_response_1-class_{split}_{str_dateTime}.txt")

dict_files = MatFileLoader(config.DATA_PATH_DEFECT_DETECTION,
                           files_to_load_str_pattern=config.FILES_TO_LOAD_STR_PATTERNS,
                           fileName_output=fileName_output).load_files()

dataExtractor = DataExtractor(fileName_output)
data, labels, file_names = dataExtractor.transform_loaded_data_to_ML_timeseries_data(dict_files, scale_wrt_hammer=True)

ml = MachineLearningOneClassPipeline(one_class_classifiers, fileName_output,
                                     scale_data=False, balance_data=False)

list_of_auc = []
nbr_runs = 10
test_file_names = None

for i in range(nbr_runs):

    if split == "random":
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5)
    elif split == "split-by-specimen":

        normal_data_for_training = ["normal_", ]
        for i in range(1, 11):
            normal_data_for_training.append(f"normal{i}_")

        normal_data_for_training, _ = _shuffle_and_split(normal_data_for_training, test_size=0.5)
        train_data, test_data, train_labels, test_labels, train_file_names, test_file_names = \
            ml.train_test_split_manual_per_specimen(data, labels, file_names, normal_data_for_training)

    dict_scores_by_classifier, auc = ml.train_and_classify(train_data, test_data,
                                                      train_labels, test_labels, test_file_names,
                                                      description=f"ROCKAD. train/test split = {split}")
    list_of_auc.append(auc)

res = f"average AUC: {np.mean(list_of_auc)}"
print(res)

if fileName_output != "":
    with open(fileName_output, "a") as text_file:
        text_file.writelines("##############################################\n")
        text_file.writelines(f"AUC (area under the ROC curve), averaged over all {nbr_runs} runs:\n")

        text_file.writelines(f"mean value: {np.mean(list_of_auc)}\n")
        text_file.writelines(f"standard deviation: {np.std(list_of_auc)}\n")
        text_file.writelines(f"min value: {np.min(list_of_auc)}\n")
        text_file.writelines(f"max value: {np.max(list_of_auc)}\n")

        text_file.writelines("##############################################\n")


df = dict_scores_by_classifier[one_class_classifiers[0]] # get results for ROCKAD

df.to_csv(fileName_output + '_scores.csv', index=False)


def plot_results(df):
    df_sorted = df.sort_values(by="anomaly score")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
#    print(df_sorted)
    #print(df)

    plt.scatter(df["class_label_int"], df["anomaly score"], marker='o', color='b')
    plt.xlabel("class_label_int")
    plt.ylabel("anomaly score")
    plt.title("Anomaly scores w.r.t. classes")
    plt.show()


    plt.plot(df["anomaly score"], marker='o')
    plt.ylabel("anomaly score")
    plt.title("Anomaly scores")
    plt.show()


    import seaborn as sns
#    sns.histplot(df, x="anomaly score", hue="class label", kde=True, bins=10)
#    plt.title("Anomaly scores w.r.t. classes")
#    plt.show()

    sns.kdeplot(data=df, x="anomaly score", hue="class_label_int", fill=True)
    plt.title("Anomaly scores w.r.t. classes")
    plt.show()

    sns.boxplot(x="class_label_int", y="anomaly score", data=df)
    plt.title("Anomaly scores w.r.t. classes")
    plt.show()

    #df["class label"][df["class label"] > 0] = 1
    #sns.kdeplot(data=df, x="anomaly score", hue="class label", fill=True)
    #plt.title("Anomaly scores w.r.t. classes")
    #plt.show()

    #sns.boxplot(x="class label", y="anomaly score", data=df)
    #plt.title("Anomaly scores w.r.t. classes")
    #plt.show()

plot_results(df)

