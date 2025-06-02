##################################################################
## This file contains the classes and functions to read, segments, convert the data and to train the models
## The other files use this source code
## i.e. this file itself does not execute anything if it is run

# imports

import os
import glob
from collections import Counter

import scipy.io
from scipy.signal import find_peaks
from pathlib import Path

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.manifold import TSNE

import random
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


from sklearn import metrics

# classify in the time series domain
# !pip install aeon or using conda
# https://www.aeon-toolkit.org/en/stable/examples/classification/classification.html


from aeon.classification.dictionary_based import *
from aeon.classification.interval_based import *
from aeon.classification.hybrid import *
from aeon.classification.convolution_based import *
from aeon.classification.deep_learning import *

from aeon.classification.distance_based import * # KNeighborsTimeSeriesClassifier, ElasticEnsemble
from aeon.classification.convolution_based import * # RocketClassifier
from aeon.classification.feature_based import * # Catch22Classifier, FreshPRINCEClassifier, TSFreshClassifier
from aeon.classification.shapelet_based import ShapeletTransformClassifier
from aeon.classification.sklearn import RotationForestClassifier

###################################################################################

class MyConfig:
    pass

config = MyConfig()

config.DATA_PATH = "../../data"

config.DATA_PATH_DEFECT_DETECTION = os.path.join(config.DATA_PATH, "raw_data_defect_detection")

config.DATA_PATH_DESICCATION = os.path.join(config.DATA_PATH, "raw_data_desiccation")

config.RESULTS_PATH = "../../results"


# string patterns for files to load -- checks if the filename starts with one of the specified patterns

config.FILES_TO_LOAD_STR_PATTERNS = ["normal_s1_h3",        # fabrication 1
                "cylinder_s1_h3",
                "sphere_s1_h3",
                "circle_s1_h3"]
config.FILES_TO_LOAD_STR_PATTERNS.extend(                   # fabrication 2
                            [#"normal_s1_h3", # duplicate name
                            "ssquare_s1_h3",
                            "bsquare_s1_h3",
                            "slit_s1_h3",
                            "cylinderld_s1_h3",
                            "sphereld_s1_h3"]
                            )


# fabrication 3 + 4
files_normal_fab3_fab4 = []

for i in range(1,11):           # for 2025 journal extension: new normal files ("normal1 ... normal10")
    files_normal_fab3_fab4.append(f"normal{i}_s1_h3")

config.FILES_TO_LOAD_STR_PATTERNS.extend(files_normal_fab3_fab4)


#FILES_TO_LOAD_STR_PATTERNS = [""]  # to load all files in folder

config.INDEX_SENSOR = 0
config.INDEX_HAMMER = 1

# in accordance with original Matlab code -- will extract -20 ... + 150, i.e.  SEGMENT_IDX_PRE + SEGMENT_IDX_POST + 1 data points
config.SEGMENT_IDX_PRE = 20
config.SEGMENT_IDX_POST = 150

config.SENSOR_THRESHOLD_MIN = 0.1  # in accordance with original Matlab code

config.PEAK_THRESHOLD_LOWER = 0.85
config.PEAK_THRESHOLD_UPPER = 1.15


config.CLASS_LABELS_INT_BY_NAME = {"normal" : 0,
                            "cylinder" : 1,
                            "sphere" : 2,
                            "circle" : 3,
                            "ssquare" : 4,
                            "bsquare" : 5,
                            "slit" : 6,
                            "cylinderld" : 7,
                            "sphereld": 8}


config.CLASS_LABELS_NAME_BY_INT = {value: key for key, value in config.CLASS_LABELS_INT_BY_NAME.items()}


# run classifiers in parallel
config.N_PARALLEL_JOBS = 5 # -1 to use all cores



classifiers = [
    ##### distance-based

    # most basic method to do it: 1-NN with Euclidean distance  (data needs to be scaled!!)
    KNeighborsTimeSeriesClassifier(distance="euclidean", n_neighbors=1, n_jobs=config.N_PARALLEL_JOBS),

    ###### time series feature extractors:
    TSFreshClassifier(default_fc_parameters = "efficient", verbose=True, random_state = 42), # other: "comprehensive"
    Catch22Classifier(catch24=False, n_jobs=config.N_PARALLEL_JOBS, random_state = 42),

    RandomIntervalSpectralEnsembleClassifier(n_jobs=config.N_PARALLEL_JOBS, random_state = 42), # features from frequency domain

    ### time series ensembles
    HIVECOTEV2(verbose=True, time_limit_in_minutes=2, n_jobs=config.N_PARALLEL_JOBS, random_state = 42),

    WEASEL_V2(n_jobs=config.N_PARALLEL_JOBS, random_state = 42),       # new for journal extension 2025
    IndividualBOSS(n_jobs=config.N_PARALLEL_JOBS, random_state = 42),  # new for journal extension 2025
    SummaryClassifier(n_jobs=config.N_PARALLEL_JOBS, random_state = 42),  # new for journal extension 2025
    TimeSeriesForestClassifier(n_jobs=config.N_PARALLEL_JOBS, random_state = 42),  # new for journal extension 2025
    ShapeletTransformClassifier(estimator=RotationForestClassifier(n_estimators=3), n_jobs=config.N_PARALLEL_JOBS, random_state = 42), # new for journal extension 2025

    ### ROCKET
    RocketClassifier(num_kernels=10000, n_jobs=config.N_PARALLEL_JOBS)#,
   # not used: RocketClassifier(rocket_transform="multirocket", num_kernels=10000, n_jobs=config.N_PARALLEL_JOBS)
]


classifiers_just_rocket = [
    RocketClassifier(num_kernels=10000, n_jobs=config.N_PARALLEL_JOBS)
]


# use the selected classifier ROCKET (makes experiments run much faster), otherwise test all of the above
classifiers = classifiers_just_rocket

###################################################################################

class MatFileLoader:

    def __init__(self, data_path, files_to_load_str_pattern=[""], fileName_output=""):
        """

        :param data_path: data path to load the .mat files from
        :param files_to_load_str_pattern: only loads the files starting with any of the strings in this list (default: load all)
        """
        self.data_path = data_path
        self.files_to_load_str_pattern = files_to_load_str_pattern
        self.fileName_output = fileName_output

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"MatFileLoader: data_path = {data_path}\n")
                text_file.writelines(f" files to load pattern: {files_to_load_str_pattern}\n")

    def load_files(self, file_names=None, verbose=False):
        """
        Loads all files with names in file_names
        :param file_names: loads files from folder or alternatively the list of filenames (needs to included relative or absolute path)
        :param verbose: prints details about loaded files
        :return: dictionary dict_data_by_filename with key: file name, value: data
        """
        dict_data_by_filename = {} # dictionary with filenames as keys and the matrix of hammer and sensor signal as the value

        if file_names == None:
            file_names = self.get_file_names() # get file names from folder


        # Load each .mat file and store in a dictionary with key: filename, value: data
        for mat_file in file_names:
            file_name = os.path.basename(mat_file) # get file name without path
            file_name = Path(file_name).stem  # discard file extension


            # load file and extract the two relevant rows (note 3:5 extracts row 3 and 4, excluding 5)
            ## specific comment one Matlab vs. Python code difference
            # note: Python indexes start at [0] in contrast to Matlab where indexes start at [1], hence, the difference between the two source

            if file_name in dict_data_by_filename.keys():
                dict_data_by_filename[file_name+"_new"] = scipy.io.loadmat(mat_file)['data'][3:5,:]
            else:
                dict_data_by_filename[file_name] = scipy.io.loadmat(mat_file)['data'][3:5,:]



        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Number of files loaded: {len(dict_data_by_filename)}\n")

                if(verbose == True):
                    text_file.writelines("Loaded files: \n")

                    for file_name, loaded_data in dict_data_by_filename.items():
                        text_file.writelines(file_name)
        #                print(f"Data from file '{file_name}':")
        #                print(loaded_data)
                        text_file.writelines("\n")


        return dict_data_by_filename


    def __get_subfolders_1st_level(self, root_dir):
        subfolders_1stlevel = []
        for path, dirnames, filenames in os.walk(root_dir):
            for dir in dirnames:
                subfolders_1stlevel.append(os.path.join(path, dir))

        return subfolders_1stlevel



    def get_file_names(self):
        """
        get list of all file names with the extension .mat in the data_path and its 1st level subfolders
        returns all filenames starting with any of the strings in files_to_load_str_pattern

        :return:     list of filenames found matching FILES_TO_LOAD_STR_PATTERNS (including the data_path)
        """
        selected_filenames = []

        # Find all .mat files in the directory
        all_mat_files_in_folder = glob.glob(os.path.join(self.data_path, '*.mat'))


        # Find all .mat files in the subfolders on 1st level
        subfolders = self.__get_subfolders_1st_level(self.data_path)
        for subfolder in subfolders:
            all_mat_files_in_folder.extend(glob.glob(os.path.join(subfolder, '*.mat')))


        for the_file in all_mat_files_in_folder:

            file_name = os.path.basename(the_file) # get file name without path

            if any(file_name.startswith(str_pattern) for str_pattern in self.files_to_load_str_pattern):
                selected_filenames.append(the_file)

        return selected_filenames




###################################################



class DataExtractor:

    def __init__(self, fileName_output=""):
        self.fileName_output = fileName_output


    def extract_segments_by_peak(self, dict_data_by_filename):
        """
        extracts time series segments based on identified peaks
        uses constants SEGMENT_IDX_PRE and SEGMENT_IDX_POST to determine the segment lengths
        :param dict_data_by_filename: dictionary with key: filename, data: time series data
        :return: list of extracted segments with elements per segment: [extr_segment_sensor, extr_segment_hammer, file_name]
        """

        extracted_segments = [] # matrix with one row per extracted segment
                                # sensor in [,0] , hammer signal in [,1] and filname in [,2]

        for file_name, loaded_data in dict_data_by_filename.items():

            # from reponse signal peaks (as in original Matlab code)
            peaks_indexes_signal = scipy.signal.find_peaks(loaded_data[config.INDEX_SENSOR], distance=1000)[0]  # min dist according to original Matlab code

            # from hammer signal peaks -- results are quite similar
            #peaks_indexes_signal = scipy.signal.find_peaks(loaded_data[config.INDEX_HAMMER], distance=1000)[0]  # min dist according to original Matlab code

            for idx_peak in peaks_indexes_signal:
                idx_start = idx_peak - config.SEGMENT_IDX_PRE
                idx_end = idx_peak + config.SEGMENT_IDX_POST

                #Andreas: I used the signals' length here whereas the original code uses Tlen
                if idx_start >= 0 and idx_end <= len(loaded_data[config.INDEX_SENSOR]) and idx_end <= len(loaded_data[config.INDEX_HAMMER]) :
                    if max(loaded_data[config.INDEX_SENSOR][idx_start:(idx_end+1)]) > config.SENSOR_THRESHOLD_MIN:
                        extr_segment_sensor = loaded_data[config.INDEX_SENSOR][idx_start:(idx_end+1)] # note: python excludes the end index, hence the +1
                        extr_segment_hammer = loaded_data[config.INDEX_HAMMER][idx_start:(idx_end+1)]
                        extracted_segments.append( [extr_segment_sensor, extr_segment_hammer, file_name] )

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Extracted segments: {len(extracted_segments)}\n")

        return extracted_segments



    ############################## FILTER ###################################

    def select_segments(self, segments):
        # filter segments: only use the segments where the peak of the hammer signal is within in predefined range PEAK_THRESHOLD_LOWER ... PEAK_THRESHOLD_UPPER ,
        # on the scaled hammer signal, i.e. hammer signal's that are within PEAK_THRESHOLD_LOWER ... PEAK_THRESHOLD_UPPER w.r.t. the mean value of all hammer signal peaks

        # original Matlab code:
        # dummy = mean(max(hit_comb,[],2));
        # dummy_norm = max(hit_comb,[],2)./dummy;

        max_per_segment = [max(segm[config.INDEX_HAMMER]) for segm in segments]
        scaling_factor_mean_of_maxes = np.mean(max_per_segment)
        scaled_max_values = max_per_segment / scaling_factor_mean_of_maxes

        idx_max_values_within_range = [idx for idx in range(len(scaled_max_values)) if config.PEAK_THRESHOLD_LOWER <= scaled_max_values[idx] <= config.PEAK_THRESHOLD_UPPER]

        extracted_segments_selected = [segments[idx] for idx in idx_max_values_within_range]

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Selected segments: {len(extracted_segments_selected)}\n")

        return extracted_segments_selected


    def select_segments_std(self, segments):
        # filter segments: only use the segments where the peak of the hammer signal is within in predefined range PEAK_THRESHOLD_LOWER ... PEAK_THRESHOLD_UPPER ,
        # on the scaled hammer signal, i.e. hammer signal's that are within PEAK_THRESHOLD_LOWER ... PEAK_THRESHOLD_UPPER w.r.t. the mean value of all hammer signal peaks

        # original Matlab code:
        # dummy = mean(max(hit_comb,[],2));
        # dummy_norm = max(hit_comb,[],2)./dummy;

        max_per_segment = [max(segm[config.INDEX_HAMMER]) for segm in segments]
        max_values_mean = np.mean(max_per_segment)
        max_values_std = np.std(max_per_segment)

        # just keep the ones within 2 * standard deviation
        idx_max_values_within_range = [idx for idx in range(len(max_per_segment)) if abs(max_per_segment[idx] - max_values_mean) < 2 * max_values_std ]

        extracted_segments_selected = [segments[idx] for idx in idx_max_values_within_range]

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Selected segments: {len(extracted_segments_selected)}\n")

        return extracted_segments_selected

    def extract_features_min_max_peaks(self, response, num_pks):
        """
        extract features similar to the original Matlab code (max and min peaks: values and location/index)

        :param response:
        :param num_pks:
        :return:
        """
        features = []

        for signal in response:

            peaks_indexes_signal = scipy.signal.find_peaks(signal, distance=5)[0]  # min dist according to original Matlab code
            peaks_values_signal = [signal[idx] for idx in peaks_indexes_signal]

            # find the number of max and mean values specified by the parameter num_pks
            matrix = np.array([
                peaks_indexes_signal,
                peaks_values_signal
            ])

            idx_sorted = np.argsort(matrix[1, :]) # sort by peak values in asc order
            idx_sorted = idx_sorted[::-1] # reverse to descending order, i.e. max values first

            sorted_matrix = matrix[:, idx_sorted] # extract new matrix in sorted order

            feature_vector = []
            feature_vector.extend(sorted_matrix[1, 0:num_pks])  # max peak values
            feature_vector.extend(sorted_matrix[1, -num_pks:])  # min peak values (Note: negative indexes start at the last element, with [-1] = last
            feature_vector.extend(sorted_matrix[0, 0:num_pks])  # time points (indexes) of max peaks
            feature_vector.extend(sorted_matrix[0, -num_pks:])  # time points (indexes) of min peaks

            features.append(feature_vector)

        return np.array(features)


    def transform_loaded_data_to_ML_timeseries_data(self, dict_data_by_filename, scale_wrt_hammer = False, multivariate = False, class_name_is_prefix = True):
        """
        transform the loaded data such that it can be passed to ML models, i.e. separates data and labels
         1) extracts univariate signals or uses full multivariate set
         2) creates numeric class labels based on file names (requires filenames to include "_")

        uses constant CLASS_LABELS_INT_BY_NAME to determine the mapping of names to numeric class labels

        :param dict_data_by_filename: loaded data
                multivariate: if True, extracts a multivariate time series of hammer and sensor signal
        :return: one numpy arrays with the data, one numpy array with numeric class labels
        """
        segments = self.extract_segments_by_peak(dict_data_by_filename)
        return self.transform_to_ML_timeseries_data(segments, scale_wrt_hammer, multivariate, class_name_is_prefix)


    def transform_to_ML_timeseries_data(self, segments, scale_wrt_hammer = False, multivariate = False, class_name_is_prefix = True):
        """
        transform the time series data such that it can be passed to ML models, i.e. separates data and labels
         1) extracts univariate signals or uses full multivariate set
         2) creates numeric class labels based on file names (requires filenames to include "_")

        uses constant CLASS_LABELS_INT_BY_NAME to determine the mapping of names to numeric class labels

        :param segments: segmented time series
                multivariate: if True, extracts a multivariate time series of hammer and sensor signal
        :return: one numpy arrays with the data, one numpy array with numeric class labels
        """

        separator_char = "_"  # NOTE: requires filenames to include "_"

        data_ts = []
        labels = []
        file_names = []

        for segm in segments:

            scale_factor = 1

            if scale_wrt_hammer == True:
                scale_factor = max(abs(segm[config.INDEX_HAMMER]))

            if multivariate == True:
                data_ts.append([segm[config.INDEX_HAMMER] / scale_factor, segm[config.INDEX_SENSOR] / scale_factor])
            else:
                data_ts.append(segm[config.INDEX_SENSOR] / scale_factor)

            class_name = ""
            if class_name_is_prefix:
                class_name = segm[2].split(separator_char)[0]# substring before first separator_char is used as class name
                class_name = class_name.rstrip('0123456789')  # if class name ends with number, remove the number
                # for example "normal1" => "normal"
            else:
                class_name = segm[2].split(separator_char)[-2]# substring before last separator_char is used as class name
                class_name = class_name.rstrip('0123456789')  # if class name ends with number, remove the number

            file_names.append(segm[2]) # file name
            labels.append(config.CLASS_LABELS_INT_BY_NAME.get(class_name)) # numeric class labels

        data_ts = np.array(data_ts)
        labels = np.array(labels)
        file_names = np.array(file_names)

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Multivariate: {multivariate}\n")
                text_file.writelines(f"Data: {data_ts.shape}, labels: {labels.shape}\n")
                text_file.writelines(f"Scale with respect to hammer signal: {scale_wrt_hammer}\n")

        return data_ts, labels, file_names

##############################################################################################

class DataDescriptor:

    def plot_loaded_data(self, dict_data_by_filename, filename_prefix="", row2plot=config.INDEX_SENSOR):

        plt.figure(figsize=(10, 6))


        ## specific comment one Matlab vs. Python code difference
        # 4th row in Matlab notation (where index starts at 1) corresponds to row [0] here => TODO: double-check
        for file_name, loaded_data in dict_data_by_filename.items():
            if file_name.startswith(filename_prefix) or filename_prefix == "":
                plt.plot(loaded_data[row2plot, :], label=file_name)

        # Add labels and title
        plt.xlabel('index of data point')
        plt.ylabel('signal')
        plt.title(filename_prefix)
        # plt.legend()
        plt.show()

    def describe_ML_prepared_data(self, data, labels):

        print(f"Dimensionality of data: {data.shape}")
        print(f"Labels: {np.unique(labels)}")
        print(f"Data items per class: {[labels.tolist().count(lab) for lab in np.unique(labels)]}")


    def plot_ML_prepared_data(self, data, labels, select_label=""):

        COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#f61600", "#7834a9",
                  "#17becf", "#684427", "#fa5deb", "#17becf", "#17becf"]

        if max(labels > len(COLORS)):
            raise Exception('Sorry, I am running out of colors here when trying to assign colors to class labels... ')

        plt.figure(figsize=(10, 6))

        for i in range(0, len(labels)):

            dat = data[i]
            label = labels[i]

            if label == select_label or select_label == "":
                plt.plot(dat, c=COLORS[label], label=label)

        # Add labels and title
        plt.xlabel('index of data point')
        plt.ylabel('signal')
        plt.title(select_label)
        # plt.legend()
        plt.show()


    def plot_TSNE(self, data, labels):

        tsne = TSNE(n_components=2, learning_rate='auto', method='exact')
        X_embedded = tsne.fit_transform(data)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ticks=[0, 1, 2], label='Classes')
        plt.title('t-SNE visualization')
        plt.xlabel('t-SNE dim 1')
        plt.ylabel('t-SNE dim 2')
        plt.legend()
        plt.show()


##############################################################################################

class MachineLearningPipeline:

    def __init__(self, classifiers, fileName_output="", scale_data = False, balance_data = False):
        self.classifiers = classifiers
        self.fileName_output = fileName_output
        self.scale_data = scale_data
        self.balance_data = balance_data

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Machine learning pipeline configuration: scale_date = {scale_data},"
                                     f" balance_data = {balance_data}\n")


    def balance_2_class_data_set_by_downsampling(self, data, labels):
        count = Counter(labels.tolist())
        label_majority_class, _ = count.most_common()[0]

        print(f"Downsampling class {label_majority_class}")

        data_majority = data[labels == label_majority_class]
        data_minority = data[labels != label_majority_class]

        labels_majority = labels[labels == label_majority_class]
        labels_minority = labels[labels != label_majority_class]


        data_majority_downsampled, labels_majority_downsampled = resample(data_majority, labels_majority,
                                                                          replace=False,
                                                                          n_samples=len(data_minority),
                                                                          random_state=123)

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Balance data (downsampling): majority class orig: {len(labels_majority)}, after downsampling: {len(labels_majority_downsampled)}\n")


        return np.concatenate([data_majority_downsampled, data_minority]), \
            np.concatenate([labels_majority_downsampled, labels_minority])


    def train_test_split_per_specimen(self, data, labels, file_names, file_names_for_train_data, verbose=True):

        assert(len(data) == len(labels) == len(file_names))
        print(f"train_test_split_manual_per_specimen: len(data) {len(data)}")

        file_names_for_train_data = tuple(file_names_for_train_data)

        idx_train = np.array([file_name.startswith(file_names_for_train_data) for file_name in file_names])
        idx_test = np.logical_not(idx_train)

        if verbose == True:
            print(f"idx_train: {sum(idx_train)}")
            print(f"idx_test: {sum(idx_test)}")

        assert( (sum(idx_train) + sum(idx_test)) == len(data) )

        # from boolean to numeric indices
        idx_train = np.where(idx_train == True)
        idx_test = np.where(idx_test == True)

        train_data = np.take(data, idx_train, axis=0)[0]
        test_data = np.take(data, idx_test, axis=0)[0]

        if verbose == True:
            print(f"train_test_split_manual_per_specimen: len(train_data) {len(train_data)}")
            print(f"train_test_split_manual_per_specimen: len(test_data) {len(test_data)}")

        train_labels = np.take(labels, idx_train, axis=0)[0]
        test_labels = np.take(labels, idx_test, axis=0)[0]

        if file_names is not None:
            train_file_names = np.take(file_names, idx_train, axis=0)[0]
            test_file_names = np.take(file_names, idx_test, axis=0)[0]
            return train_data, test_data, train_labels, test_labels, train_file_names, test_file_names

        return train_data, test_data, train_labels, test_labels



    def train_and_classify(self, train_data, test_data, train_labels, test_labels, test_file_names=None, description="train_and_classify"):
        """
        trains and tests all classifiers passed while creating the object
        writes outputs to file
        :return: the mean value of the balanced accuracy, averaged over all classifiers
        """
        if self.balance_data == True:
            # balance data by downsampling majority class
            train_data, train_labels = self.balance_2_class_data_set_by_downsampling(train_data, train_labels)
            test_data, test_labels = self.balance_2_class_data_set_by_downsampling(test_data, test_labels)

        if self.scale_data == True:
            # scaling row-wise, i.e. per time series
            scaler = StandardScaler()
            train_data = np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in train_data])
            test_data = np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in test_data])

        str_data_info = f"################# {description} #################\n" + \
                        f"Training classifier on {len(train_labels)} data items with dimension of training data: {train_data.shape}\n" + \
                        f"Testing classifier on {len(test_labels)} data items\n"
        print(str_data_info)


        str_latex_table = ""
        avg_acc_bal = 0

        results_by_classifier = {} # create empty dictionary

        ### train the classifiers and obtain results
        for clf in self.classifiers:
            print(f"Training and classification with: {clf}")

            random.seed(42) # to get reproducible results
            clf.fit(train_data, train_labels)
            y_pred = clf.predict(test_data)
            acc_bal = metrics.balanced_accuracy_score(test_labels, y_pred)
            avg_acc_bal += acc_bal

            recall_scores = metrics.recall_score(test_labels, y_pred, average = None) * 100
            str_latex_table += f"{clf} & {round(acc_bal*100)} \% & {round(np.min(recall_scores))} \% - {round(np.max(recall_scores))} \%  \n"

            if test_file_names is not None:
                labels_and_results = list(zip(test_labels, test_file_names, y_pred))  # scores per class
                results_by_classifier[str(clf)] = pd.DataFrame(data=labels_and_results,
                                                         columns=["class label", "file name", "prediction"])

            res = self.__get_results_as_string(test_labels, y_pred)
            res = f"\nClassifier: {clf}\n" + str_data_info + res

            if self.fileName_output != "":
                with open(self.fileName_output, "a") as text_file:
                    text_file.writelines(res)

                with open(self.fileName_output, "r") as text_file:
                    file_content = text_file.read()
                    print(file_content)

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                if test_file_names is not None:
                    text_file.writelines("test_file_names:\n")
                #    for fn in np.unique(test_file_names):
                 #       text_file.writelines(f"{fn}\n")

                text_file.writelines("\n\n ### LATEX TABLE FORMAT ### \n")
                text_file.writelines(str_latex_table)
                text_file.writelines("\n\n")

            print("Results written to: " + self.fileName_output)

        avg_acc_bal = avg_acc_bal / len(self.classifiers)

        return results_by_classifier, avg_acc_bal


    def __get_results_as_string(self, lab, pred):
        """
        returns a formatted string with classification results, based on the class labels and a classier's predictions
        :param lab: class labels
        :param pred: predictions
        :return: results as string
        """
        str_res = ""
        acc = metrics.balanced_accuracy_score(lab, pred)
        cr = metrics.classification_report(lab, pred)
        cm = metrics.confusion_matrix(lab, pred)
        str_res += f"\nBalanced accuracy: {round(acc,2)}"
        str_res += f"\n{cr}\n"
        str_res += "\nConfusion matrix:\n"
        str_res += str(cm)
        str_res += "\n-------------------------------\n"

        return str_res


###################################################################


class MachineLearningOneClassPipeline:

    def __init__(self, classifiers, fileName_output="", scale_data = True, balance_data = False):
        self.classifiers = classifiers
        self.fileName_output = fileName_output
        self.scale_data = scale_data
        self.balance_data = balance_data

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Machine learning ONE-CLASS pipeline configuration: scale_date = {scale_data},"
                                     f" balance_data = {balance_data}\n")


    def balance_2_class_data_set_by_downsampling(self, data, labels):
        count = Counter(labels.tolist())
        label_majority_class, _ = count.most_common()[0]

        print(f"Downsampling class {label_majority_class}")

        data_majority = data[labels == label_majority_class]
        data_minority = data[labels != label_majority_class]

        labels_majority = labels[labels == label_majority_class]
        labels_minority = labels[labels != label_majority_class]


        data_majority_downsampled, labels_majority_downsampled = resample(data_majority, labels_majority,
                                                                          replace=False,
                                                                          n_samples=len(data_minority),
                                                                          random_state=123)

        if self.fileName_output != "":
            with open(self.fileName_output, "a") as text_file:
                text_file.writelines(f"Balance data (downsampling): majority class orig: {len(labels_majority)}, after downsampling: {len(labels_majority_downsampled)}\n")


        return np.concatenate([data_majority_downsampled, data_minority]), \
            np.concatenate([labels_majority_downsampled, labels_minority])


    def train_test_split_sklearn(self, data, labels):

        train_data, test_data, train_labels, test_labels = train_test_split(
                                   data, labels, test_size=0.2, random_state=123)

        return train_data, test_data, train_labels, test_labels


    def train_test_split_manual_shuffle(self, data, labels, file_names=None):
        indices = np.arange(len(data))
        random.seed(42)
        random.shuffle(indices)

        data_shuffled = data[indices]
        labels_shuffled = labels[indices]
        file_names_shuffled = file_names[indices]

        test_size = 0.2
        split_index = int(len(data) * (1 - test_size))
        train_data = data_shuffled[:split_index]
        test_data = data_shuffled[split_index:]
        train_labels = labels_shuffled[:split_index]
        test_labels = labels_shuffled[split_index:]

        if file_names is not None:
            train_file_names = file_names_shuffled[:split_index]
            test_file_names = file_names_shuffled[split_index:]
            return train_data, test_data, train_labels, test_labels, train_file_names, test_file_names

        return train_data, test_data, train_labels, test_labels


    def train_test_split_manual_per_specimen(self, data, labels, file_names, file_names_for_train_data, verbose=True):

        assert(len(data) == len(labels) == len(file_names))
        print(f"train_test_split_manual_per_specimen: len(data) {len(data)}")

        idx_normal = np.array(labels == config.CLASS_LABELS_INT_BY_NAME["normal"])
        idx_anomalies = np.logical_not(idx_normal)

        file_names_for_train_data = tuple(file_names_for_train_data)

        idx_normal_train_data = np.array([file_name.startswith(file_names_for_train_data) for file_name in file_names])
        idx_normal_test_data = np.logical_not(idx_normal_train_data)

        # construct train and test set from indices
        idx_train = idx_normal_train_data
        idx_test = np.logical_or(idx_normal_test_data, idx_anomalies)

        if verbose == True:
            print(f"idx_normal: {sum(idx_normal)}")
            print(f"idx_normal_train_data: {sum(idx_normal_train_data)}")
            print(f"idx_normal_test_data: {sum(idx_normal_test_data)}")
            print(f"idx_train: {sum(idx_train)}")
            print(f"idx_test: {sum(idx_test)}")

        assert( (sum(idx_train) + sum(idx_test)) == len(data) )

        # from boolean to numeric indices
        idx_train = np.where(idx_train == True)
        idx_test = np.where(idx_test == True)

        train_data = np.take(data, idx_train, axis=0)[0]
        test_data = np.take(data, idx_test, axis=0)[0]

        if verbose == True:
            print(f"train_test_split_manual_per_specimen: len(train_data) {len(train_data)}")
            print(f"train_test_split_manual_per_specimen: len(test_data) {len(test_data)}")

        train_labels = np.take(labels, idx_train, axis=0)[0]
        test_labels = np.take(labels, idx_test, axis=0)[0]

        if file_names is not None:
            train_file_names = np.take(file_names, idx_train, axis=0)[0]
            test_file_names = np.take(file_names, idx_test, axis=0)[0]
            return train_data, test_data, train_labels, test_labels, train_file_names, test_file_names

        return train_data, test_data, train_labels, test_labels



    def train_and_classify(self, train_data, test_data, train_labels, test_labels, test_file_names = None, description="train_and_classify"):

        if self.balance_data == True:
            # balance data by downsampling majority class
            test_data, test_labels = self.balance_2_class_data_set_by_downsampling(test_data, test_labels)

        if self.scale_data == True:
            # scaling row-wise, i.e. per time series
            scaler = StandardScaler()
            train_data = np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in train_data])
            test_data = np.array([scaler.fit_transform(row.reshape(-1, 1)).flatten() for row in test_data])

        # store original class labels
        test_labels_with_classes_int = copy.deepcopy(test_labels)
        test_labels_with_class_names = [config.CLASS_LABELS_NAME_BY_INT[x] for x in test_labels_with_classes_int]

        # convert the class labels to two classes: normal == 1 and anomaly == -1
        # to be compatible with sklearn outlier detectors (sklearn uses +1 for normal and -1 for anomaly/outlier)
        train_labels = np.array(
            [1 if lab == 0 else -1 for lab in train_labels])  # set all classes other than normal to -1
        test_labels = np.array([1 if lab == 0 else -1 for lab in test_labels])  # set all classes other than normal to -1

        # train on normal dataset (i.e. remove all classes other than normal from the training data)
        train_data = train_data[train_labels == 1]
        train_labels = train_labels[train_labels == 1]

        str_data_info = f"################# {description} #################\n" + \
                        f"Training anomaly detector on {len(train_labels)} data items with dimension of training data: {train_data.shape}\n"
        print(str_data_info)

        # convert data to 3D np aray such that sktime interpretes each row as one time series
        # https://github.com/sktime/sktime/blob/main/examples/AA_datatypes_and_datasets.ipynb
        # 3D np.ndarray, of shape (n_instances, n_variables, n_timepoints)
        #TODO: check if this can be dropped when using aeon implementation
        train_data = train_data[:, np.newaxis, :]
        test_data = test_data[:, np.newaxis, :]

        scores_by_classifier = {} # create empty dictionary

        ### train the classifiers and obtain results
        for clf in self.classifiers:
            print(f"Training and classification with: {clf}")

            clf.fit(train_data)
            scores = clf.predict_proba(test_data)  # Predict anomaly scores

            if test_file_names is not None:
                labels_and_scores = list(zip(test_labels_with_classes_int, test_labels_with_class_names, test_file_names, scores))  # scores per class
                scores_by_classifier[clf] = pd.DataFrame(data=labels_and_scores,
                                                         columns=["class_label_int", "class_label_name", "file name", "anomaly score"])
            else:
                labels_and_scores = list(zip(test_labels_with_classes_int, test_labels_with_class_names, scores)) # scores per class
                scores_by_classifier[clf] = pd.DataFrame(data=labels_and_scores, columns=["class_label_int", "class_label_name", "anomaly score"])

            fpr, tpr, thresholds = metrics.roc_curve(test_labels, scores, pos_label=-1)
            roc_auc = metrics.auc(fpr, tpr)

            print(description)
            res = f"\nClassifier: {clf}\n" + str_data_info + f"AUC: {roc_auc}\n\n"

            # convert anomaly scores to crisp class labels normal/anomaly
            # here using simple nearest neighbor one-class implementation  (other one-class classifiers are possible!)
            #from NearestNeighborOCC import NearestNeighborOCC
            #clf_nnOCC = NearestNeighborOCC()
            #clf_nnOCC.fit(clf.predict_proba(train_data))
            #y_pred = clf_nnOCC.predict(clf.predict_proba(test_data))
            #res = res + "\n one-class classification using nearest neighbors on anomaly scores: \n" + self.__get_results_as_string(test_labels, y_pred)

            if self.fileName_output != "":
                with open(self.fileName_output, "a") as text_file:
                    text_file.writelines(res)
                print("Results written to: " + self.fileName_output)

                with open(self.fileName_output, "r") as text_file:
                    file_content = text_file.read()
                    print(file_content)

            return scores_by_classifier, roc_auc


    def __get_results_as_string(self, lab, pred):
        """
        returns a formatted string with classification results, based on the class labels and a classier's predictions
        :param lab: class labels
        :param pred: predictions
        :return: results as string
        """
        str_res = ""
        acc = metrics.accuracy_score(lab, pred)
        cr = metrics.classification_report(lab, pred)
        cm = metrics.confusion_matrix(lab, pred)
        str_res += f"\nOverall accuracy: {round(acc,2)}"
        str_res += f"\n{cr}\n"
        str_res += "\nConfusion matrix:\n"
        str_res += str(cm)
        str_res += "\n-------------------------------\n"

        return str_res


