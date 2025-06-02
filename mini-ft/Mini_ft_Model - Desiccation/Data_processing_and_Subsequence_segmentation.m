%{ 
---------------------------------------------------------------------------
       Mini-ft model data preparation and subsequence segmentation
---------------------------------------------------------------------------
The purpose of this file is to process (scaling) the raw data and to
extract out the isolated response subsequences. The output of these files
are isolated response subsequences (171 long). Specifically, this file
covers the initial parts of the Mini-ft model, please refer to Figure 1 of
our paper with the caption ,"AI-powered smart manufacture of
biopolymer-bound composites using simple vibration-based testing."

Data preparation:
Including both isolating subsequences and pre-processing the data to
account for variatoons in the strength of the impulse hammer. 

Development of the Mini-ft model: Specifically, we extract all of the 171
human-engineered features, which in this case correspond to all of the
response amplitudes of an isolated subsequence. We save all of these
subsequences as a .mat file, which are used as input to a seperate code
that performs dimensionality reduction using t-SNE. 

Steps in this code: 

1. Load in the raw data based on unqiue tapping location and location where
the sensor is placed. Extract out the 4th and 5th row of each raw data
file, as they contain the z-axis acceleration (out of the specimen) and the
associated impulse hammer recordings. 

2. Now we can isolate the subsequences by using a helper function called
" findpeaks_c.m". Each inidividual subsequnece for a given sensor location
and tapping location is saved as an array. Where the rows represent an
individual subsequence, while the columns represent the isolated responses
of the subsequence. 

3. Save the isolated subsequences representing the isolated subsequences of
the desiccation data for dimensionality reduction in the "dimensionality
reduction" file

%} 
%% Common Options in Analysis

% Initial Plotting commands
clc 
clear all
close all

% Load in one of the files to get the time length
Fs=2048; 
TimeT=length(load("Raw_Data\Platform_1_FRONT_2hr_1.mat").data)/Fs;
space= TimeT/(length(load("Raw_Data\Platform_1_FRONT_2hr_1.mat").data)-1);
time= 0:(space):TimeT;

%% 1. Load in the data based on the separate entities 

%{
For the raw data each file contains 5 rows of measurements, of which rows
2-5 contains useful information: 

Row 2. x-direction acceleration
Row 3. y-direction acceleration
Row 4. z-direction acceleration
Row 5. Time recording (up to 10s)

Therefore, for our purposes we care about the last two rows of each data
file. As we have five 10-second time-series recordings for each tapping
location and sensor pair, we will have to load in 5 separate data files, of
which we want the last two rows. 

For our experiment we have according to Supplementary Table 1, we have
provided data of a single sample, with desiccation data from 2-7 hours.

%}

%%%%%%%%%%%%%%%%%%%%% Load all the data for Platform 1 %%%%%%%%%%%%%%%%%%%%%%%

% Front 
data_Plat1_F_2hr= cat(1,load("Raw_Data\Platform_1_FRONT_2hr_1.mat").data,...
   load("Raw_Data\Platform_1_FRONT_2hr_2.mat").data,load("Raw_Data\Platform_1_FRONT_2hr_3.mat").data,...
   load("Raw_Data\Platform_1_FRONT_2hr_4.mat").data,load("Raw_Data\Platform_1_FRONT_2hr_5.mat").data);
data_Plat1_F_2hr= data_Plat1_F_2hr([4,5,9,10,14,15,19,20,24,25],:);

data_Plat1_F_3hr= cat(1,load("Raw_Data\Platform_1_FRONT_3hr_1.mat").data,...
   load("Raw_Data\Platform_1_FRONT_3hr_2.mat").data,load("Raw_Data\Platform_1_FRONT_3hr_3.mat").data,...
   load("Raw_Data\Platform_1_FRONT_3hr_4.mat").data,load("Raw_Data\Platform_1_FRONT_3hr_5.mat").data);
data_Plat1_F_3hr= data_Plat1_F_3hr([4,5,9,10,14,15,19,20,24,25],:);

data_Plat1_F_4hr= cat(1,load("Raw_Data\Platform_1_FRONT_4hr_1.mat").data,...
   load("Raw_Data\Platform_1_FRONT_4hr_2.mat").data,load("Raw_Data\Platform_1_FRONT_4hr_3.mat").data,...
   load("Raw_Data\Platform_1_FRONT_4hr_4.mat").data,load("Raw_Data\Platform_1_FRONT_4hr_5.mat").data);
data_Plat1_F_4hr= data_Plat1_F_4hr([4,5,9,10,14,15,19,20,24,25],:);

data_Plat1_F_5hr= cat(1,load("Raw_Data\Platform_1_FRONT_5hr_1.mat").data,...
   load("Raw_Data\Platform_1_FRONT_5hr_2.mat").data,load("Raw_Data\Platform_1_FRONT_5hr_3.mat").data,...
   load("Raw_Data\Platform_1_FRONT_5hr_4.mat").data,load("Raw_Data\Platform_1_FRONT_5hr_5.mat").data);
data_Plat1_F_5hr= data_Plat1_F_5hr([4,5,9,10,14,15,19,20,24,25],:);

data_Plat1_F_6hr= cat(1,load("Raw_Data\Platform_1_FRONT_6hr_1.mat").data,...
   load("Raw_Data\Platform_1_FRONT_6hr_2.mat").data,load("Raw_Data\Platform_1_FRONT_6hr_3.mat").data,...
   load("Raw_Data\Platform_1_FRONT_6hr_4.mat").data,load("Raw_Data\Platform_1_FRONT_6hr_5.mat").data);
data_Plat1_F_6hr= data_Plat1_F_6hr([4,5,9,10,14,15,19,20,24,25],:);

data_Plat1_F_7hr= cat(1,load("Raw_Data\Platform_1_FRONT_7hr_1.mat").data,...
   load("Raw_Data\Platform_1_FRONT_7hr_2.mat").data,load("Raw_Data\Platform_1_FRONT_7hr_3.mat").data,...
   load("Raw_Data\Platform_1_FRONT_7hr_4.mat").data,load("Raw_Data\Platform_1_FRONT_7hr_5.mat").data);
data_Plat1_F_7hr= data_Plat1_F_7hr([4,5,9,10,14,15,19,20,24,25],:);


%% 2. Get the hammer hits and responses
% Now for each one, isolate the subsequences using the findpeaks function
% and then scale/normalize the data based on the max height of all the
% hammer recordings. 

Tlen = length(time);

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Platform 1 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Front 
response_platform1_F_2hr= []; hammerhits_platform1_F_2hr= [];
for i=1:5 
I_response = data_Plat1_F_2hr(2*i-1,:);
I_hammer= data_Plat1_F_2hr(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_platform1_F_2hr= cat(1, response_platform1_F_2hr, response_p);
hammerhits_platform1_F_2hr= cat(1, hammerhits_platform1_F_2hr, hammerhits_p);
response_platform1_F_2hr( ~any(response_platform1_F_2hr,2), : ) = [];  %rows
hammerhits_platform1_F_2hr( ~any(hammerhits_platform1_F_2hr,2), : ) = [];  %rows
end

response_platform1_F_3hr= []; hammerhits_platform1_F_3hr= [];
for i=1:5 
I_response = data_Plat1_F_3hr(2*i-1,:);
I_hammer= data_Plat1_F_3hr(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_platform1_F_3hr= cat(1, response_platform1_F_3hr, response_p);
hammerhits_platform1_F_3hr= cat(1, hammerhits_platform1_F_3hr, hammerhits_p);
response_platform1_F_3hr( ~any(response_platform1_F_3hr,2), : ) = [];  %rows
hammerhits_platform1_F_3hr( ~any(hammerhits_platform1_F_3hr,2), : ) = [];  %rows
end

response_platform1_F_4hr= []; hammerhits_platform1_F_4hr= [];
for i=1:5 
I_response = data_Plat1_F_4hr(2*i-1,:);
I_hammer= data_Plat1_F_4hr(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_platform1_F_4hr= cat(1, response_platform1_F_4hr, response_p);
hammerhits_platform1_F_4hr= cat(1, hammerhits_platform1_F_4hr, hammerhits_p);
response_platform1_F_4hr( ~any(response_platform1_F_4hr,2), : ) = [];  %rows
hammerhits_platform1_F_4hr( ~any(hammerhits_platform1_F_4hr,2), : ) = [];  %rows
end

response_platform1_F_5hr= []; hammerhits_platform1_F_5hr= [];
for i=1:5 
I_response = data_Plat1_F_5hr(2*i-1,:);
I_hammer= data_Plat1_F_5hr(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_platform1_F_5hr= cat(1, response_platform1_F_5hr, response_p);
hammerhits_platform1_F_5hr= cat(1, hammerhits_platform1_F_5hr, hammerhits_p);
response_platform1_F_5hr( ~any(response_platform1_F_5hr,2), : ) = [];  %rows
hammerhits_platform1_F_5hr( ~any(hammerhits_platform1_F_5hr,2), : ) = [];  %rows
end

response_platform1_F_6hr= []; hammerhits_platform1_F_6hr= [];
for i=1:5 
I_response = data_Plat1_F_6hr(2*i-1,:);
I_hammer= data_Plat1_F_6hr(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_platform1_F_6hr= cat(1, response_platform1_F_6hr, response_p);
hammerhits_platform1_F_6hr= cat(1, hammerhits_platform1_F_6hr, hammerhits_p);
response_platform1_F_6hr( ~any(response_platform1_F_6hr,2), : ) = [];  %rows
hammerhits_platform1_F_6hr( ~any(hammerhits_platform1_F_6hr,2), : ) = [];  %rows
end

response_platform1_F_7hr= []; hammerhits_platform1_F_7hr= [];
for i=1:5 
I_response = data_Plat1_F_7hr(2*i-1,:);
I_hammer= data_Plat1_F_7hr(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_platform1_F_7hr= cat(1, response_platform1_F_7hr, response_p);
hammerhits_platform1_F_7hr= cat(1, hammerhits_platform1_F_7hr, hammerhits_p);
response_platform1_F_7hr( ~any(response_platform1_F_7hr,2), : ) = [];  %rows
hammerhits_platform1_F_7hr( ~any(hammerhits_platform1_F_7hr,2), : ) = [];  %rows
end


%% 3. Find the middle range of hammer force and only care about those
% Now save the extracted subsequences for dimensionality reduction on the
% second python file. After running the python file to perform
% dimensionality reduction come back and run the section below

% Normal specimens
save('features_response_front_2hr.mat', 'response_platform1_F_2hr')
save('features_response_front_3hr.mat', 'response_platform1_F_3hr')
save('features_response_front_4hr.mat', 'response_platform1_F_4hr')
save('features_response_front_5hr.mat', 'response_platform1_F_5hr')
save('features_response_front_6hr.mat', 'response_platform1_F_6hr')
save('features_response_front_7hr.mat', 'response_platform1_F_7hr')

%% 4. Perform the regression
clc;
clear all;
close all;

% Load the dataset
data = load('tsne_encoded_response_front.mat').tsne_encoded_features;  % Adjust variable name if needed

% Separate features and response
X = data(:, 1:2);  % First two columns are features
y = data(:, 3);    % Third column is response

% Settings
numRuns = 100;

% Preallocate metric arrays
maeVals = zeros(numRuns, 1);
mapeVals = zeros(numRuns, 1);
mseVals = zeros(numRuns, 1);
r2Vals = zeros(numRuns, 1);

% Collect all predictions and actuals
allYTest = [];
allYPred = [];

% Loop through 100 random splits
for i = 1:numRuns
    % 80/20 random split
    cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
    XTrain = X(training(cv), :);
    yTrain = y(training(cv));
    XTest = X(test(cv), :);
    yTest = y(test(cv));

    % Train GPR model
    gprMdl = fitrgp(XTrain, yTrain, 'KernelFunction', 'exponential');

    % Predict
    yPred = predict(gprMdl, XTest);

    % Store predictions
    allYTest = [allYTest; yTest];
    allYPred = [allYPred; yPred];

    % Metrics
    errors = yTest - yPred;
    maeVals(i) = mean(abs(errors));
    mapeVals(i) = mean(abs(errors ./ yTest)) * 100;
    mseVals(i) = mean(errors.^2);
    
    % R² (1 - SS_res / SS_tot)
    ss_res = sum(errors.^2);
    ss_tot = sum((yTest - mean(yTest)).^2);
    r2Vals(i) = 1 - ss_res / ss_tot;
end

% ==== Summary Statistics ====
fprintf('\nSummary of 100 GPR Runs (Exponential Kernel):\n');
fprintf('Average MAE : %.4f\n', mean(maeVals));
fprintf('Average MAPE: %.2f%%\n', mean(mapeVals));
fprintf('Average MSE : %.4f\n', mean(mseVals));
fprintf('Average R²  : %.4f\n', mean(r2Vals));

% ==== Overall metrics from all combined predictions ====
errors_all = allYTest - allYPred;
mae_all = mean(abs(errors_all));
mape_all = mean(abs(errors_all ./ allYTest)) * 100;
mse_all = mean(errors_all.^2);
r2_all = 1 - sum(errors_all.^2) / sum((allYTest - mean(allYTest)).^2);

fprintf('\nOverall Metrics from Combined Predictions:\n');
fprintf('MAE  : %.4f\n', mae_all);
fprintf('MAPE : %.2f%%\n', mape_all);
fprintf('MSE  : %.4f\n', mse_all);
fprintf('R²   : %.4f\n', r2_all);

% ==== Plots ====

% MSE distribution
figure;
histogram(mseVals, 15);
xlabel('MSE');
ylabel('Frequency');
title('MSE Distribution over 100 Runs');
grid on;

% Predicted vs Actual
figure;
scatter(allYTest, allYPred, 10, 'filled');
xlabel('Actual');
ylabel('Predicted');
title('Predicted vs. Actual (100 GPR Runs)');
grid on;
