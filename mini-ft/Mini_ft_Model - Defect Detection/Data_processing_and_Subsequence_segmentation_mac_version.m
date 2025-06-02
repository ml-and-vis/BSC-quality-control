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

3. For all the hammer recordings we find the maximum impulse strength
applied to all the specimens, and scale up the responses of all the other
subsequences so that the subsequences have the same impulse applied to the
specimen. 

4. Now with the maximum hammer strength we can scale up the responses to
account for variations in the impulse strength applied on the specimens

5. We find the maximum scaled response amplitude and normalize all the
responses so that they lie between 0 and 1 in amplitude

6. Save the isolated and scaled subsequences for further processing in the
associated python file "Dimensionality reduction", from where you can
calculate reduced dimensions/tSNE features which are then visualized using
the later file titled "TSNE_Visualization"

%} 
%% Common Options in Analysis

clc 
clear all
close all

Fs=2048; % The recorded sampling frequency

% Define the time series based on one of the files. Using any of the raw
% data files will be sufficient
TimeT=length(load("Raw_Data/normal_s1_h2_1.mat").data)/Fs;
space= TimeT/(length(load("Raw_Data/normal_s1_h2_1.mat").data)-1);
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

For our experiment we have according to Supplementary Table 1, 19 samples
here for defect detection, as broken down here. 

- 11 normal specimens
- 8 unique defect types
%}

rows_to_select = [4,5,9,10,14,15,19,20,24,25];

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal %%%%%%%%%%%%%%%%%%%%%%%

% because normal_s1_h2_4 was a little shorter we will make some padding at
% the end of the recording with zeroes to make sure that we can concatenate
% all the data in the dataframe
data_fix_1 = cat(2, load("Raw_Data/normal_s1_h2_4.mat").data, (zeros(5,20500-19680)));
data_normal_S1_h2= cat(1,load("Raw_Data/normal_s1_h2_1.mat").data,...
   load("Raw_Data/normal_s1_h2_2.mat").data,load("Raw_Data/normal_s1_h2_3.mat").data,...
   data_fix_1,load("Raw_Data/normal_s1_h2_5.mat").data);
data_normal0_S1_h2= data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h3= cat(1,load("Raw_Data/normal_s1_h3_1.mat").data,...
   load("Raw_Data/normal_s1_h3_2.mat").data,load("Raw_Data/normal_s1_h3_3.mat").data,...
   load("Raw_Data/normal_s1_h3_4.mat").data,load("Raw_Data/normal_s1_h3_5.mat").data);
data_normal0_S1_h3= data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h5= cat(1,load("Raw_Data/normal_s1_h5_1.mat").data,...
   load("Raw_Data/normal_s1_h5_2.mat").data,load("Raw_Data/normal_s1_h5_3.mat").data,...
   load("Raw_Data/normal_s1_h5_4.mat").data,load("Raw_Data/normal_s1_h5_5.mat").data);
data_normal0_S1_h5= data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25],:);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal1 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2 = cat(1, load('Raw_Data/normal1_s1_h2_1.mat').data, ...
    load('Raw_Data/normal1_s1_h2_2.mat').data, load('Raw_Data/normal1_s1_h2_3.mat').data, ...
    load('Raw_Data/normal1_s1_h2_4.mat').data, load('Raw_Data/normal1_s1_h2_6.mat').data);
data_normal1_S1_h2 = data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h3 = cat(1, load('Raw_Data/normal1_s1_h3_1.mat').data, ...
    load('Raw_Data/normal1_s1_h3_2.mat').data, load('Raw_Data/normal1_s1_h3_3.mat').data, ...
    load('Raw_Data/normal1_s1_h3_4.mat').data, load('Raw_Data/normal1_s1_h3_5.mat').data);
data_normal1_S1_h3 = data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h5 = cat(1, load('Raw_Data/normal1_s1_h5_1.mat').data, ...
    load('Raw_Data/normal1_s1_h5_2.mat').data, load('Raw_Data/normal1_s1_h5_3.mat').data, ...
    load('Raw_Data/normal1_s1_h5_4.mat').data, load('Raw_Data/normal1_s1_h5_5.mat').data);
data_normal1_S1_h5 = data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal2 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2 = cat(1, load('Raw_Data/normal2_s1_h2_1.mat').data, ...
    load('Raw_Data/normal2_s1_h2_2.mat').data, load('Raw_Data/normal2_s1_h2_3.mat').data, ...
    load('Raw_Data/normal2_s1_h2_4.mat').data, load('Raw_Data/normal2_s1_h2_5.mat').data);
data_normal2_S1_h2 = data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h3 = cat(1, load('Raw_Data/normal2_s1_h3_1.mat').data, ...
    load('Raw_Data/normal2_s1_h3_2.mat').data, load('Raw_Data/normal2_s1_h3_3.mat').data, ...
    load('Raw_Data/normal2_s1_h3_4.mat').data, load('Raw_Data/normal2_s1_h3_5.mat').data);
data_normal2_S1_h3 = data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h5 = cat(1, load('Raw_Data/normal2_s1_h5_1.mat').data, ...
    load('Raw_Data/normal2_s1_h5_2.mat').data, load('Raw_Data/normal2_s1_h5_3.mat').data, ...
    load('Raw_Data/normal2_s1_h5_4.mat').data, load('Raw_Data/normal2_s1_h5_5.mat').data);
data_normal2_S1_h5 = data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal3 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2 = cat(1, load('Raw_Data/normal3_s1_h2_1.mat').data, ...
    load('Raw_Data/normal3_s1_h2_2.mat').data, load('Raw_Data/normal3_s1_h2_3.mat').data, ...
    load('Raw_Data/normal3_s1_h2_4.mat').data, load('Raw_Data/normal3_s1_h2_5.mat').data);
data_normal3_S1_h2 = data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h3 = cat(1, load('Raw_Data/normal3_s1_h3_1.mat').data, ...
    load('Raw_Data/normal3_s1_h3_2.mat').data, load('Raw_Data/normal3_s1_h3_3.mat').data, ...
    load('Raw_Data/normal3_s1_h3_4.mat').data, load('Raw_Data/normal3_s1_h3_5.mat').data);
data_normal3_S1_h3 = data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h5 = cat(1, load('Raw_Data/normal3_s1_h5_1.mat').data, ...
    load('Raw_Data/normal3_s1_h5_2.mat').data, load('Raw_Data/normal3_s1_h5_3.mat').data, ...
    load('Raw_Data/normal3_s1_h5_4.mat').data, load('Raw_Data/normal3_s1_h5_5.mat').data);
data_normal3_S1_h5 = data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal4 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2 = cat(1, load('Raw_Data/normal4_s1_h2_1.mat').data, ...
    load('Raw_Data/normal4_s1_h2_2.mat').data, load('Raw_Data/normal4_s1_h2_3.mat').data, ...
    load('Raw_Data/normal4_s1_h2_4.mat').data, load('Raw_Data/normal4_s1_h2_5.mat').data);
data_normal4_S1_h2 = data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h3 = cat(1, load('Raw_Data/normal4_s1_h3_1.mat').data, ...
    load('Raw_Data/normal4_s1_h3_2.mat').data, load('Raw_Data/normal4_s1_h3_3.mat').data, ...
    load('Raw_Data/normal4_s1_h3_4.mat').data, load('Raw_Data/normal4_s1_h3_5.mat').data);
data_normal4_S1_h3 = data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h5 = cat(1, load('Raw_Data/normal4_s1_h5_1.mat').data, ...
    load('Raw_Data/normal4_s1_h5_2.mat').data, load('Raw_Data/normal4_s1_h5_3.mat').data, ...
    load('Raw_Data/normal4_s1_h5_4.mat').data, load('Raw_Data/normal4_s1_h5_5.mat').data);
data_normal4_S1_h5 = data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal5 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2 = cat(1, load('Raw_Data/normal5_s1_h2_1.mat').data, ...
    load('Raw_Data/normal5_s1_h2_2.mat').data, load('Raw_Data/normal5_s1_h2_3.mat').data, ...
    load('Raw_Data/normal5_s1_h2_4.mat').data, load('Raw_Data/normal5_s1_h2_5.mat').data);
data_normal5_S1_h2 = data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h3 = cat(1, load('Raw_Data/normal5_s1_h3_1.mat').data, ...
    load('Raw_Data/normal5_s1_h3_2.mat').data, load('Raw_Data/normal5_s1_h3_3.mat').data, ...
    load('Raw_Data/normal5_s1_h3_4.mat').data, load('Raw_Data/normal5_s1_h3_5.mat').data);
data_normal5_S1_h3 = data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_normal_S1_h5 = cat(1, load('Raw_Data/normal5_s1_h5_1.mat').data, ...
    load('Raw_Data/normal5_s1_h5_2.mat').data, load('Raw_Data/normal5_s1_h5_3.mat').data, ...
    load('Raw_Data/normal5_s1_h5_4.mat').data, load('Raw_Data/normal5_s1_h5_5.mat').data);
data_normal5_S1_h5 = data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25], :);


%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal6 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2= cat(1,load("Raw_Data/normal6_s1_h2_1.mat").data,...
   load("Raw_Data/normal6_s1_h2_2.mat").data,load("Raw_Data/normal6_s1_h2_3.mat").data,...
   load("Raw_Data/normal6_s1_h2_4.mat").data,load("Raw_Data/normal6_s1_h2_6.mat").data);
data_normal6_S1_h2= data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h3= cat(1,load("Raw_Data/normal6_s1_h3_1.mat").data,...
   load("Raw_Data/normal6_s1_h3_2.mat").data,load("Raw_Data/normal6_s1_h3_3.mat").data,...
   load("Raw_Data/normal6_s1_h3_4.mat").data,load("Raw_Data/normal6_s1_h3_5.mat").data);
data_normal6_S1_h3= data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h5= cat(1,load("Raw_Data/normal6_s1_h5_1.mat").data,...
   load("Raw_Data/normal6_s1_h5_2.mat").data,load("Raw_Data/normal6_s1_h5_3.mat").data,...
   load("Raw_Data/normal6_s1_h5_4.mat").data,load("Raw_Data/normal6_s1_h5_5.mat").data);
data_normal6_S1_h5= data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25],:);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal7 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2= cat(1,load("Raw_Data/normal7_s1_h2_1.mat").data,...
   load("Raw_Data/normal7_s1_h2_2.mat").data,load("Raw_Data/normal7_s1_h2_3.mat").data,...
   load("Raw_Data/normal7_s1_h2_4.mat").data,load("Raw_Data/normal7_s1_h2_5.mat").data);
data_normal7_S1_h2= data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h3= cat(1,load("Raw_Data/normal7_s1_h3_1.mat").data,...
   load("Raw_Data/normal7_s1_h3_2.mat").data,load("Raw_Data/normal7_s1_h3_3.mat").data,...
   load("Raw_Data/normal7_s1_h3_4.mat").data,load("Raw_Data/normal7_s1_h3_5.mat").data);
data_normal7_S1_h3= data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h5= cat(1,load("Raw_Data/normal7_s1_h5_1.mat").data,...
   load("Raw_Data/normal7_s1_h5_2.mat").data,load("Raw_Data/normal7_s1_h5_3.mat").data,...
   load("Raw_Data/normal7_s1_h5_4.mat").data,load("Raw_Data/normal7_s1_h5_5.mat").data);
data_normal7_S1_h5= data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25],:);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal8 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2= cat(1,load("Raw_Data/normal8_s1_h2_1.mat").data,...
   load("Raw_Data/normal8_s1_h2_2.mat").data,load("Raw_Data/normal8_s1_h2_3.mat").data,...
   load("Raw_Data/normal8_s1_h2_4.mat").data,load("Raw_Data/normal8_s1_h2_5.mat").data);
data_normal8_S1_h2= data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h3= cat(1,load("Raw_Data/normal8_s1_h3_1.mat").data,...
   load("Raw_Data/normal8_s1_h3_2.mat").data,load("Raw_Data/normal8_s1_h3_3.mat").data,...
   load("Raw_Data/normal8_s1_h3_4.mat").data,load("Raw_Data/normal8_s1_h3_5.mat").data);
data_normal8_S1_h3= data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h5= cat(1,load("Raw_Data/normal8_s1_h5_1.mat").data,...
   load("Raw_Data/normal8_s1_h5_2.mat").data,load("Raw_Data/normal8_s1_h5_3.mat").data,...
   load("Raw_Data/normal8_s1_h5_4.mat").data,load("Raw_Data/normal8_s1_h5_5.mat").data);
data_normal8_S1_h5= data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25],:);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal9 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2= cat(1,load("Raw_Data/normal9_s1_h2_1.mat").data,...
   load("Raw_Data/normal9_s1_h2_2.mat").data,load("Raw_Data/normal9_s1_h2_3.mat").data,...
   load("Raw_Data/normal9_s1_h2_4.mat").data,load("Raw_Data/normal9_s1_h2_5.mat").data);
data_normal9_S1_h2= data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h3= cat(1,load("Raw_Data/normal9_s1_h3_1.mat").data,...
   load("Raw_Data/normal9_s1_h3_2.mat").data,load("Raw_Data/normal9_s1_h3_3.mat").data,...
   load("Raw_Data/normal9_s1_h3_4.mat").data,load("Raw_Data/normal9_s1_h3_5.mat").data);
data_normal9_S1_h3= data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h5= cat(1,load("Raw_Data/normal9_s1_h5_1.mat").data,...
   load("Raw_Data/normal9_s1_h5_2.mat").data,load("Raw_Data/normal9_s1_h5_3.mat").data,...
   load("Raw_Data/normal9_s1_h5_4.mat").data,load("Raw_Data/normal9_s1_h5_5.mat").data);
data_normal9_S1_h5= data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25],:);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Normal10 %%%%%%%%%%%%%%%%%%%%%%%

data_normal_S1_h2= cat(1,load("Raw_Data/normal10_s1_h2_1.mat").data,...
   load("Raw_Data/normal10_s1_h2_2.mat").data,load("Raw_Data/normal10_s1_h2_3.mat").data,...
   load("Raw_Data/normal10_s1_h2_4.mat").data,load("Raw_Data/normal10_s1_h2_5.mat").data);
data_normal10_S1_h2= data_normal_S1_h2([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h3= cat(1,load("Raw_Data/normal10_s1_h3_2.mat").data,...
   load("Raw_Data/normal10_s1_h3_3.mat").data,load("Raw_Data/normal10_s1_h3_4.mat").data,...
   load("Raw_Data/normal10_s1_h3_5.mat").data,load("Raw_Data/normal10_s1_h3_6.mat").data);
data_normal10_S1_h3= data_normal_S1_h3([4,5,9,10,14,15,19,20,24,25],:);

data_normal_S1_h5= cat(1,load("Raw_Data/normal10_s1_h5_1.mat").data,...
   load("Raw_Data/normal10_s1_h5_2.mat").data,load("Raw_Data/normal10_s1_h5_3.mat").data,...
   load("Raw_Data/normal10_s1_h5_4.mat").data,load("Raw_Data/normal10_s1_h5_5.mat").data);
data_normal10_S1_h5= data_normal_S1_h5([4,5,9,10,14,15,19,20,24,25],:);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Sphere hd %%%%%%%%%%%%%%%%%%%%%%%

data_sphere_S1_h2 = cat(1, load("Raw_Data/sphere_s1_h2_1.mat").data, ...
   load("Raw_Data/sphere_s1_h2_2.mat").data, load("Raw_Data/sphere_s1_h2_3.mat").data, ...
   load("Raw_Data/sphere_s1_h2_4.mat").data, load("Raw_Data/sphere_s1_h2_5.mat").data);
data_sphere_S1_h2 = data_sphere_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_sphere_S1_h3 = cat(1, load("Raw_Data/sphere_s1_h3_1.mat").data, ...
   load("Raw_Data/sphere_s1_h3_2.mat").data, load("Raw_Data/sphere_s1_h3_3.mat").data, ...
   load("Raw_Data/sphere_s1_h3_4.mat").data, load("Raw_Data/sphere_s1_h3_5.mat").data);
data_sphere_S1_h3 = data_sphere_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_sphere_S1_h5 = cat(1, load("Raw_Data/sphere_s1_h5_1.mat").data, ...
   load("Raw_Data/sphere_s1_h5_2.mat").data, load("Raw_Data/sphere_s1_h5_3.mat").data, ...
   load("Raw_Data/sphere_s1_h5_4.mat").data, load("Raw_Data/sphere_s1_h5_5.mat").data);
data_sphere_S1_h5 = data_sphere_S1_h5([4,5,9,10,14,15,19,20,24,25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Cylinder hd %%%%%%%%%%%%%%%%%%%%%%%

% Location S1
data_cylinder_S1_h2 = cat(1, load("Raw_Data/cylinder_s1_h2_1.mat").data, ...
   load("Raw_Data/cylinder_s1_h2_2.mat").data, load("Raw_Data/cylinder_s1_h2_3.mat").data, ...
   load("Raw_Data/cylinder_s1_h2_4.mat").data, load("Raw_Data/cylinder_s1_h2_5.mat").data);
data_cylinder_S1_h2 = data_cylinder_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_cylinder_S1_h3 = cat(1, load("Raw_Data/cylinder_s1_h3_1.mat").data, ...
   load("Raw_Data/cylinder_s1_h3_2.mat").data, load("Raw_Data/cylinder_s1_h3_3.mat").data, ...
   load("Raw_Data/cylinder_s1_h3_4.mat").data, load("Raw_Data/cylinder_s1_h3_5.mat").data);
data_cylinder_S1_h3 = data_cylinder_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_cylinder_S1_h5 = cat(1, load("Raw_Data/cylinder_s1_h5_1.mat").data, ...
   load("Raw_Data/cylinder_s1_h5_2.mat").data, load("Raw_Data/cylinder_s1_h5_3.mat").data, ...
   load("Raw_Data/cylinder_s1_h5_4.mat").data, load("Raw_Data/cylinder_s1_h5_5.mat").data);
data_cylinder_S1_h5 = data_cylinder_S1_h5([4,5,9,10,14,15,19,20,24,25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for Circle %%%%%%%%%%%%%%%%%%%%%%%

% Location S1
data_circle_S1_h2 = cat(1, load("Raw_Data/circle_s1_h2_1.mat").data, ...
   load("Raw_Data/circle_s1_h2_2.mat").data, load("Raw_Data/circle_s1_h2_3.mat").data, ...
   load("Raw_Data/circle_s1_h2_4.mat").data, load("Raw_Data/circle_s1_h2_5.mat").data);
data_circle_S1_h2 = data_circle_S1_h2([4,5,9,10,14,15,19,20,24,25], :);

data_circle_S1_h3 = cat(1, load("Raw_Data/circle_s1_h3_1.mat").data, ...
   load("Raw_Data/circle_s1_h3_2.mat").data, load("Raw_Data/circle_s1_h3_3.mat").data, ...
   load("Raw_Data/circle_s1_h3_4.mat").data, load("Raw_Data/circle_s1_h3_5.mat").data);
data_circle_S1_h3 = data_circle_S1_h3([4,5,9,10,14,15,19,20,24,25], :);

data_circle_S1_h5 = cat(1, load("Raw_Data/circle_s1_h5_1.mat").data, ...
   load("Raw_Data/circle_s1_h5_2.mat").data, load("Raw_Data/circle_s1_h5_3.mat").data, ...
   load("Raw_Data/circle_s1_h5_4.mat").data, load("Raw_Data/circle_s1_h5_5.mat").data);
data_circle_S1_h5 = data_circle_S1_h5([4,5,9,10,14,15,19,20,24,25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for sphereld %%%%%%%%%%%%%%%%%%%%%%%

% Location S1
data_sphereld_S1_h2 = cat(1, load("Raw_Data/sphereld_s1_h2_1.mat").data, ...
    load("Raw_Data/sphereld_s1_h2_2.mat").data, load("Raw_Data/sphereld_s1_h2_3.mat").data, ...
    load("Raw_Data/sphereld_s1_h2_4.mat").data, load("Raw_Data/sphereld_s1_h2_5.mat").data);
data_sphereld_S1_h2 = data_sphereld_S1_h2([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_fix_1 = cat(2, load("Raw_Data/sphereld_s1_h3_5.mat").data, (zeros(5, 20500 - 15375)));
data_sphereld_S1_h3 = cat(1, load("Raw_Data/sphereld_s1_h3_1.mat").data, ...
    load("Raw_Data/sphereld_s1_h3_2.mat").data, load("Raw_Data/sphereld_s1_h3_3.mat").data, ...
    load("Raw_Data/sphereld_s1_h3_4.mat").data, data_fix_1);
data_sphereld_S1_h3 = data_sphereld_S1_h3([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_sphereld_S1_h5 = cat(1, load("Raw_Data/sphereld_s1_h5_1.mat").data, ...
    load("Raw_Data/sphereld_s1_h5_2.mat").data, load("Raw_Data/sphereld_s1_h5_3.mat").data, ...
    load("Raw_Data/sphereld_s1_h5_4.mat").data, load("Raw_Data/sphereld_s1_h5_5.mat").data);
data_sphereld_S1_h5 = data_sphereld_S1_h5([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for cylinderld %%%%%%%%%%%%%%%%%%%%%%%

% Location S1
data_cylinderld_S1_h2 = cat(1, load("Raw_Data/cylinderld_s1_h2_1.mat").data, ...
    load("Raw_Data/cylinderld_s1_h2_2.mat").data, load("Raw_Data/cylinderld_s1_h2_3.mat").data, ...
    load("Raw_Data/cylinderld_s1_h2_4.mat").data, load("Raw_Data/cylinderld_s1_h2_5.mat").data);
data_cylinderld_S1_h2 = data_cylinderld_S1_h2([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_cylinderld_S1_h3 = cat(1, load("Raw_Data/cylinderld_s1_h3_1.mat").data, ...
    load("Raw_Data/cylinderld_s1_h3_2.mat").data, load("Raw_Data/cylinderld_s1_h3_3.mat").data, ...
    load("Raw_Data/cylinderld_s1_h3_4.mat").data, load("Raw_Data/cylinderld_s1_h3_5.mat").data);
data_cylinderld_S1_h3 = data_cylinderld_S1_h3([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_cylinderld_S1_h5 = cat(1, load("Raw_Data/cylinderld_s1_h5_1.mat").data, ...
    load("Raw_Data/cylinderld_s1_h5_2.mat").data, load("Raw_Data/cylinderld_s1_h5_3.mat").data, ...
    load("Raw_Data/cylinderld_s1_h5_4.mat").data, load("Raw_Data/cylinderld_s1_h5_5.mat").data);
data_cylinderld_S1_h5 = data_cylinderld_S1_h5([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for slit %%%%%%%%%%%%%%%%%%%%%%%

% Location S1
data_slit_S1_h2 = cat(1, load("Raw_Data/slit_s1_h2_1.mat").data, ...
    load("Raw_Data/slit_s1_h2_2.mat").data, load("Raw_Data/slit_s1_h2_3.mat").data, ...
    load("Raw_Data/slit_s1_h2_4.mat").data, load("Raw_Data/slit_s1_h2_5.mat").data);
data_slit_S1_h2 = data_slit_S1_h2([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_slit_S1_h3 = cat(1, load("Raw_Data/slit_s1_h3_1.mat").data, ...
    load("Raw_Data/slit_s1_h3_2.mat").data, load("Raw_Data/slit_s1_h3_3.mat").data, ...
    load("Raw_Data/slit_s1_h3_4.mat").data, load("Raw_Data/slit_s1_h3_5.mat").data);
data_slit_S1_h3 = data_slit_S1_h3([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_slit_S1_h5 = cat(1, load("Raw_Data/slit_s1_h5_1.mat").data, ...
    load("Raw_Data/slit_s1_h5_2.mat").data, load("Raw_Data/slit_s1_h5_3.mat").data, ...
    load("Raw_Data/slit_s1_h5_4.mat").data, load("Raw_Data/slit_s1_h5_5.mat").data);
data_slit_S1_h5 = data_slit_S1_h5([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for ssquare %%%%%%%%%%%%%%%%%%%%%%%

% Location S1
data_ssquare_S1_h2 = cat(1, load("Raw_Data/ssquare_s1_h2_1.mat").data, ...
    load("Raw_Data/ssquare_s1_h2_2.mat").data, load("Raw_Data/ssquare_s1_h2_3.mat").data, ...
    load("Raw_Data/ssquare_s1_h2_4.mat").data, load("Raw_Data/ssquare_s1_h2_5.mat").data);
data_ssquare_S1_h2 = data_ssquare_S1_h2([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_fix_1 = cat(2, load("Raw_Data/ssquare_s1_h3_1.mat").data, (zeros(5, 20500 - 20295)));
data_ssquare_S1_h3 = cat(1, data_fix_1, ...
    load("Raw_Data/ssquare_s1_h3_2.mat").data, load("Raw_Data/ssquare_s1_h3_3.mat").data, ...
    load("Raw_Data/ssquare_s1_h3_4.mat").data, load("Raw_Data/ssquare_s1_h3_5.mat").data);
data_ssquare_S1_h3 = data_ssquare_S1_h3([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_ssquare_S1_h5 = cat(1, load("Raw_Data/ssquare_s1_h5_1.mat").data, ...
    load("Raw_Data/ssquare_s1_h5_2.mat").data, load("Raw_Data/ssquare_s1_h5_3.mat").data, ...
    load("Raw_Data/ssquare_s1_h5_4.mat").data, load("Raw_Data/ssquare_s1_h5_5.mat").data);
data_ssquare_S1_h5 = data_ssquare_S1_h5([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

%%%%%%%%%%%%%%%%%%%%% Load all the data for bsquare %%%%%%%%%%%%%%%%%%%%%%%

% Location S1
data_bsquare_S1_h2 = cat(1, load("Raw_Data/bsquare_s1_h2_1.mat").data, ...
    load("Raw_Data/bsquare_s1_h2_2.mat").data, load("Raw_Data/bsquare_s1_h2_3.mat").data, ...
    load("Raw_Data/bsquare_s1_h2_4.mat").data, load("Raw_Data/bsquare_s1_h2_5.mat").data);
data_bsquare_S1_h2 = data_bsquare_S1_h2([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_bsquare_S1_h3 = cat(1, load("Raw_Data/bsquare_s1_h3_1.mat").data, ...
    load("Raw_Data/bsquare_s1_h3_2.mat").data, load("Raw_Data/bsquare_s1_h3_3.mat").data, ...
    load("Raw_Data/bsquare_s1_h3_4.mat").data, load("Raw_Data/bsquare_s1_h3_5.mat").data);
data_bsquare_S1_h3 = data_bsquare_S1_h3([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);

data_bsquare_S1_h5 = cat(1, load("Raw_Data/bsquare_s1_h5_1.mat").data, ...
    load("Raw_Data/bsquare_s1_h5_2.mat").data, load("Raw_Data/bsquare_s1_h5_3.mat").data, ...
    load("Raw_Data/bsquare_s1_h5_4.mat").data, load("Raw_Data/bsquare_s1_h5_5.mat").data);
data_bsquare_S1_h5 = data_bsquare_S1_h5([4, 5, 9, 10, 14, 15, 19, 20, 24, 25], :);


%% 2. Get the hammer hits and responses (Subsequence generation)
% Now for each one, isolate the subsequences using the findpeaks function

Tlen = length(time);

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal0 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal0_S1_h2= []; hammerhits_normal0_S1_h2= [];
for i=1:5 
I_response = data_normal0_S1_h2(2*i-1,:);
I_hammer= data_normal0_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal0_S1_h2= cat(1, response_normal0_S1_h2, response_p);
hammerhits_normal0_S1_h2= cat(1, hammerhits_normal0_S1_h2, hammerhits_p);
response_normal0_S1_h2( ~any(response_normal0_S1_h2,2), : ) = [];  %rows
hammerhits_normal0_S1_h2( ~any(hammerhits_normal0_S1_h2,2), : ) = [];  %rows
end

response_normal0_S1_h3= []; hammerhits_normal0_S1_h3= [];
for i=1:5 
I_response = data_normal0_S1_h3(2*i-1,:);
I_hammer= data_normal0_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal0_S1_h3= cat(1, response_normal0_S1_h3, response_p);
hammerhits_normal0_S1_h3= cat(1, hammerhits_normal0_S1_h3, hammerhits_p);
response_normal0_S1_h3( ~any(response_normal0_S1_h3,2), : ) = [];  %rows
hammerhits_normal0_S1_h3( ~any(hammerhits_normal0_S1_h3,2), : ) = [];  %rows
end

response_normal0_S1_h5= []; hammerhits_normal0_S1_h5= [];
for i=1:5 
I_response = data_normal0_S1_h5(2*i-1,:);
I_hammer= data_normal0_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal0_S1_h5= cat(1, response_normal0_S1_h5, response_p);
hammerhits_normal0_S1_h5= cat(1, hammerhits_normal0_S1_h5, hammerhits_p);
response_normal0_S1_h5( ~any(response_normal0_S1_h5,2), : ) = [];  %rows
hammerhits_normal0_S1_h5( ~any(hammerhits_normal0_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal1 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal1_S1_h2= []; hammerhits_normal1_S1_h2= [];
for i=1:5 
I_response = data_normal1_S1_h2(2*i-1,:);
I_hammer= data_normal1_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal1_S1_h2= cat(1, response_normal1_S1_h2, response_p);
hammerhits_normal1_S1_h2= cat(1, hammerhits_normal1_S1_h2, hammerhits_p);
response_normal1_S1_h2( ~any(response_normal1_S1_h2,2), : ) = [];  %rows
hammerhits_normal1_S1_h2( ~any(hammerhits_normal1_S1_h2,2), : ) = [];  %rows
end

response_normal1_S1_h3= []; hammerhits_normal1_S1_h3= [];
for i=1:5 
I_response = data_normal1_S1_h3(2*i-1,:);
I_hammer= data_normal1_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal1_S1_h3= cat(1, response_normal1_S1_h3, response_p);
hammerhits_normal1_S1_h3= cat(1, hammerhits_normal1_S1_h3, hammerhits_p);
response_normal1_S1_h3( ~any(response_normal1_S1_h3,2), : ) = [];  %rows
hammerhits_normal1_S1_h3( ~any(hammerhits_normal1_S1_h3,2), : ) = [];  %rows
end

response_normal1_S1_h5= []; hammerhits_normal1_S1_h5= [];
for i=1:5 
I_response = data_normal1_S1_h5(2*i-1,:);
I_hammer= data_normal1_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal1_S1_h5= cat(1, response_normal1_S1_h5, response_p);
hammerhits_normal1_S1_h5= cat(1, hammerhits_normal1_S1_h5, hammerhits_p);
response_normal1_S1_h5( ~any(response_normal1_S1_h5,2), : ) = [];  %rows
hammerhits_normal1_S1_h5( ~any(hammerhits_normal1_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal2 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal2_S1_h2= []; hammerhits_normal2_S1_h2= [];
for i=1:5 
I_response = data_normal2_S1_h2(2*i-1,:);
I_hammer= data_normal2_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal2_S1_h2= cat(1, response_normal2_S1_h2, response_p);
hammerhits_normal2_S1_h2= cat(1, hammerhits_normal2_S1_h2, hammerhits_p);
response_normal2_S1_h2( ~any(response_normal2_S1_h2,2), : ) = [];  %rows
hammerhits_normal2_S1_h2( ~any(hammerhits_normal2_S1_h2,2), : ) = [];  %rows
end

response_normal2_S1_h3= []; hammerhits_normal2_S1_h3= [];
for i=1:5 
I_response = data_normal2_S1_h3(2*i-1,:);
I_hammer= data_normal2_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal2_S1_h3= cat(1, response_normal2_S1_h3, response_p);
hammerhits_normal2_S1_h3= cat(1, hammerhits_normal2_S1_h3, hammerhits_p);
response_normal2_S1_h3( ~any(response_normal2_S1_h3,2), : ) = [];  %rows
hammerhits_normal2_S1_h3( ~any(hammerhits_normal2_S1_h3,2), : ) = [];  %rows
end

response_normal2_S1_h5= []; hammerhits_normal2_S1_h5= [];
for i=1:5 
I_response = data_normal2_S1_h5(2*i-1,:);
I_hammer= data_normal2_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal2_S1_h5= cat(1, response_normal2_S1_h5, response_p);
hammerhits_normal2_S1_h5= cat(1, hammerhits_normal2_S1_h5, hammerhits_p);
response_normal2_S1_h5( ~any(response_normal2_S1_h5,2), : ) = [];  %rows
hammerhits_normal2_S1_h5( ~any(hammerhits_normal2_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal3 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal3_S1_h2= []; hammerhits_normal3_S1_h2= [];
for i=1:5 
I_response = data_normal3_S1_h2(2*i-1,:);
I_hammer= data_normal3_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal3_S1_h2= cat(1, response_normal3_S1_h2, response_p);
hammerhits_normal3_S1_h2= cat(1, hammerhits_normal3_S1_h2, hammerhits_p);
response_normal3_S1_h2( ~any(response_normal3_S1_h2,2), : ) = [];  %rows
hammerhits_normal3_S1_h2( ~any(hammerhits_normal3_S1_h2,2), : ) = [];  %rows
end

response_normal3_S1_h3= []; hammerhits_normal3_S1_h3= [];
for i=1:5 
I_response = data_normal3_S1_h3(2*i-1,:);
I_hammer= data_normal3_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal3_S1_h3= cat(1, response_normal3_S1_h3, response_p);
hammerhits_normal3_S1_h3= cat(1, hammerhits_normal3_S1_h3, hammerhits_p);
response_normal3_S1_h3( ~any(response_normal3_S1_h3,2), : ) = [];  %rows
hammerhits_normal3_S1_h3( ~any(hammerhits_normal3_S1_h3,2), : ) = [];  %rows
end

response_normal3_S1_h5= []; hammerhits_normal3_S1_h5= [];
for i=1:5 
I_response = data_normal3_S1_h5(2*i-1,:);
I_hammer= data_normal3_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal3_S1_h5= cat(1, response_normal3_S1_h5, response_p);
hammerhits_normal3_S1_h5= cat(1, hammerhits_normal3_S1_h5, hammerhits_p);
response_normal3_S1_h5( ~any(response_normal3_S1_h5,2), : ) = [];  %rows
hammerhits_normal3_S1_h5( ~any(hammerhits_normal3_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal4 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal4_S1_h2= []; hammerhits_normal4_S1_h2= [];
for i=1:5 
I_response = data_normal4_S1_h2(2*i-1,:);
I_hammer= data_normal4_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal4_S1_h2= cat(1, response_normal4_S1_h2, response_p);
hammerhits_normal4_S1_h2= cat(1, hammerhits_normal4_S1_h2, hammerhits_p);
response_normal4_S1_h2( ~any(response_normal4_S1_h2,2), : ) = [];  %rows
hammerhits_normal4_S1_h2( ~any(hammerhits_normal4_S1_h2,2), : ) = [];  %rows
end

response_normal4_S1_h3= []; hammerhits_normal4_S1_h3= [];
for i=1:5 
I_response = data_normal4_S1_h3(2*i-1,:);
I_hammer= data_normal4_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal4_S1_h3= cat(1, response_normal4_S1_h3, response_p);
hammerhits_normal4_S1_h3= cat(1, hammerhits_normal4_S1_h3, hammerhits_p);
response_normal4_S1_h3( ~any(response_normal4_S1_h3,2), : ) = [];  %rows
hammerhits_normal4_S1_h3( ~any(hammerhits_normal4_S1_h3,2), : ) = [];  %rows
end

response_normal4_S1_h5= []; hammerhits_normal4_S1_h5= [];
for i=1:5 
I_response = data_normal4_S1_h5(2*i-1,:);
I_hammer= data_normal4_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal4_S1_h5= cat(1, response_normal4_S1_h5, response_p);
hammerhits_normal4_S1_h5= cat(1, hammerhits_normal4_S1_h5, hammerhits_p);
response_normal4_S1_h5( ~any(response_normal4_S1_h5,2), : ) = [];  %rows
hammerhits_normal4_S1_h5( ~any(hammerhits_normal4_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal5 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal5_S1_h2= []; hammerhits_normal5_S1_h2= [];
for i=1:5 
I_response = data_normal5_S1_h2(2*i-1,:);
I_hammer= data_normal5_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal5_S1_h2= cat(1, response_normal5_S1_h2, response_p);
hammerhits_normal5_S1_h2= cat(1, hammerhits_normal5_S1_h2, hammerhits_p);
response_normal5_S1_h2( ~any(response_normal5_S1_h2,2), : ) = [];  %rows
hammerhits_normal5_S1_h2( ~any(hammerhits_normal5_S1_h2,2), : ) = [];  %rows
end

response_normal5_S1_h3= []; hammerhits_normal5_S1_h3= [];
for i=1:5 
I_response = data_normal5_S1_h3(2*i-1,:);
I_hammer= data_normal5_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal5_S1_h3= cat(1, response_normal5_S1_h3, response_p);
hammerhits_normal5_S1_h3= cat(1, hammerhits_normal5_S1_h3, hammerhits_p);
response_normal5_S1_h3( ~any(response_normal5_S1_h3,2), : ) = [];  %rows
hammerhits_normal5_S1_h3( ~any(hammerhits_normal5_S1_h3,2), : ) = [];  %rows
end

response_normal5_S1_h5= []; hammerhits_normal5_S1_h5= [];
for i=1:5 
I_response = data_normal5_S1_h5(2*i-1,:);
I_hammer= data_normal5_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal5_S1_h5= cat(1, response_normal5_S1_h5, response_p);
hammerhits_normal5_S1_h5= cat(1, hammerhits_normal5_S1_h5, hammerhits_p);
response_normal5_S1_h5( ~any(response_normal5_S1_h5,2), : ) = [];  %rows
hammerhits_normal5_S1_h5( ~any(hammerhits_normal5_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal6 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal6_S1_h2= []; hammerhits_normal6_S1_h2= [];
for i=1:5 
I_response = data_normal6_S1_h2(2*i-1,:);
I_hammer= data_normal6_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal6_S1_h2= cat(1, response_normal6_S1_h2, response_p);
hammerhits_normal6_S1_h2= cat(1, hammerhits_normal6_S1_h2, hammerhits_p);
response_normal6_S1_h2( ~any(response_normal6_S1_h2,2), : ) = [];  %rows
hammerhits_normal6_S1_h2( ~any(hammerhits_normal6_S1_h2,2), : ) = [];  %rows
end

response_normal6_S1_h3= []; hammerhits_normal6_S1_h3= [];
for i=1:5 
I_response = data_normal6_S1_h3(2*i-1,:);
I_hammer= data_normal6_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal6_S1_h3= cat(1, response_normal6_S1_h3, response_p);
hammerhits_normal6_S1_h3= cat(1, hammerhits_normal6_S1_h3, hammerhits_p);
response_normal6_S1_h3( ~any(response_normal6_S1_h3,2), : ) = [];  %rows
hammerhits_normal6_S1_h3( ~any(hammerhits_normal6_S1_h3,2), : ) = [];  %rows
end

response_normal6_S1_h5= []; hammerhits_normal6_S1_h5= [];
for i=1:5 
I_response = data_normal6_S1_h5(2*i-1,:);
I_hammer= data_normal6_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal6_S1_h5= cat(1, response_normal6_S1_h5, response_p);
hammerhits_normal6_S1_h5= cat(1, hammerhits_normal6_S1_h5, hammerhits_p);
response_normal6_S1_h5( ~any(response_normal6_S1_h5,2), : ) = [];  %rows
hammerhits_normal6_S1_h5( ~any(hammerhits_normal6_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal7 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal7_S1_h2= []; hammerhits_normal7_S1_h2= [];
for i=1:5 
I_response = data_normal7_S1_h2(2*i-1,:);
I_hammer= data_normal7_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal7_S1_h2= cat(1, response_normal7_S1_h2, response_p);
hammerhits_normal7_S1_h2= cat(1, hammerhits_normal7_S1_h2, hammerhits_p);
response_normal7_S1_h2( ~any(response_normal7_S1_h2,2), : ) = [];  %rows
hammerhits_normal7_S1_h2( ~any(hammerhits_normal7_S1_h2,2), : ) = [];  %rows
end

response_normal7_S1_h3= []; hammerhits_normal7_S1_h3= [];
for i=1:5 
I_response = data_normal7_S1_h3(2*i-1,:);
I_hammer= data_normal7_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal7_S1_h3= cat(1, response_normal7_S1_h3, response_p);
hammerhits_normal7_S1_h3= cat(1, hammerhits_normal7_S1_h3, hammerhits_p);
response_normal7_S1_h3( ~any(response_normal7_S1_h3,2), : ) = [];  %rows
hammerhits_normal7_S1_h3( ~any(hammerhits_normal7_S1_h3,2), : ) = [];  %rows
end

response_normal7_S1_h5= []; hammerhits_normal7_S1_h5= [];
for i=1:5 
I_response = data_normal7_S1_h5(2*i-1,:);
I_hammer= data_normal7_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal7_S1_h5= cat(1, response_normal7_S1_h5, response_p);
hammerhits_normal7_S1_h5= cat(1, hammerhits_normal7_S1_h5, hammerhits_p);
response_normal7_S1_h5( ~any(response_normal7_S1_h5,2), : ) = [];  %rows
hammerhits_normal7_S1_h5( ~any(hammerhits_normal7_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal8 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal8_S1_h2= []; hammerhits_normal8_S1_h2= [];
for i=1:5 
I_response = data_normal8_S1_h2(2*i-1,:);
I_hammer= data_normal8_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal8_S1_h2= cat(1, response_normal8_S1_h2, response_p);
hammerhits_normal8_S1_h2= cat(1, hammerhits_normal8_S1_h2, hammerhits_p);
response_normal8_S1_h2( ~any(response_normal8_S1_h2,2), : ) = [];  %rows
hammerhits_normal8_S1_h2( ~any(hammerhits_normal8_S1_h2,2), : ) = [];  %rows
end

response_normal8_S1_h3= []; hammerhits_normal8_S1_h3= [];
for i=1:5 
I_response = data_normal8_S1_h3(2*i-1,:);
I_hammer= data_normal8_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal8_S1_h3= cat(1, response_normal8_S1_h3, response_p);
hammerhits_normal8_S1_h3= cat(1, hammerhits_normal8_S1_h3, hammerhits_p);
response_normal8_S1_h3( ~any(response_normal8_S1_h3,2), : ) = [];  %rows
hammerhits_normal8_S1_h3( ~any(hammerhits_normal8_S1_h3,2), : ) = [];  %rows
end

response_normal8_S1_h5= []; hammerhits_normal8_S1_h5= [];
for i=1:5 
I_response = data_normal8_S1_h5(2*i-1,:);
I_hammer= data_normal8_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal8_S1_h5= cat(1, response_normal8_S1_h5, response_p);
hammerhits_normal8_S1_h5= cat(1, hammerhits_normal8_S1_h5, hammerhits_p);
response_normal8_S1_h5( ~any(response_normal8_S1_h5,2), : ) = [];  %rows
hammerhits_normal8_S1_h5( ~any(hammerhits_normal8_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal9 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal9_S1_h2= []; hammerhits_normal9_S1_h2= [];
for i=1:5 
I_response = data_normal9_S1_h2(2*i-1,:);
I_hammer= data_normal9_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal9_S1_h2= cat(1, response_normal9_S1_h2, response_p);
hammerhits_normal9_S1_h2= cat(1, hammerhits_normal9_S1_h2, hammerhits_p);
response_normal9_S1_h2( ~any(response_normal9_S1_h2,2), : ) = [];  %rows
hammerhits_normal9_S1_h2( ~any(hammerhits_normal9_S1_h2,2), : ) = [];  %rows
end

response_normal9_S1_h3= []; hammerhits_normal9_S1_h3= [];
for i=1:5 
I_response = data_normal9_S1_h3(2*i-1,:);
I_hammer= data_normal9_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal9_S1_h3= cat(1, response_normal9_S1_h3, response_p);
hammerhits_normal9_S1_h3= cat(1, hammerhits_normal9_S1_h3, hammerhits_p);
response_normal9_S1_h3( ~any(response_normal9_S1_h3,2), : ) = [];  %rows
hammerhits_normal9_S1_h3( ~any(hammerhits_normal9_S1_h3,2), : ) = [];  %rows
end

response_normal9_S1_h5= []; hammerhits_normal9_S1_h5= [];
for i=1:5 
I_response = data_normal9_S1_h5(2*i-1,:);
I_hammer= data_normal9_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal9_S1_h5= cat(1, response_normal9_S1_h5, response_p);
hammerhits_normal9_S1_h5= cat(1, hammerhits_normal9_S1_h5, hammerhits_p);
response_normal9_S1_h5( ~any(response_normal9_S1_h5,2), : ) = [];  %rows
hammerhits_normal9_S1_h5( ~any(hammerhits_normal9_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Normal10 %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_normal10_S1_h2= []; hammerhits_normal10_S1_h2= [];
for i=1:5 
I_response = data_normal10_S1_h2(2*i-1,:);
I_hammer= data_normal10_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal10_S1_h2= cat(1, response_normal10_S1_h2, response_p);
hammerhits_normal10_S1_h2= cat(1, hammerhits_normal10_S1_h2, hammerhits_p);
response_normal10_S1_h2( ~any(response_normal10_S1_h2,2), : ) = [];  %rows
hammerhits_normal10_S1_h2( ~any(hammerhits_normal10_S1_h2,2), : ) = [];  %rows
end

response_normal10_S1_h3= []; hammerhits_normal10_S1_h3= [];
for i=1:5 
I_response = data_normal10_S1_h3(2*i-1,:);
I_hammer= data_normal10_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal10_S1_h3= cat(1, response_normal10_S1_h3, response_p);
hammerhits_normal10_S1_h3= cat(1, hammerhits_normal10_S1_h3, hammerhits_p);
response_normal10_S1_h3( ~any(response_normal10_S1_h3,2), : ) = [];  %rows
hammerhits_normal10_S1_h3( ~any(hammerhits_normal10_S1_h3,2), : ) = [];  %rows
end

response_normal10_S1_h5= []; hammerhits_normal10_S1_h5= [];
for i=1:5 
I_response = data_normal10_S1_h5(2*i-1,:);
I_hammer= data_normal10_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_normal10_S1_h5= cat(1, response_normal10_S1_h5, response_p);
hammerhits_normal10_S1_h5= cat(1, hammerhits_normal10_S1_h5, hammerhits_p);
response_normal10_S1_h5( ~any(response_normal10_S1_h5,2), : ) = [];  %rows
hammerhits_normal10_S1_h5( ~any(hammerhits_normal10_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Sphere hd %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_sphere_S1_h2= []; hammerhits_sphere_S1_h2= [];
for i=1:5 
I_response = data_sphere_S1_h2(2*i-1,:);
I_hammer= data_sphere_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_sphere_S1_h2= cat(1, response_sphere_S1_h2, response_p);
hammerhits_sphere_S1_h2= cat(1, hammerhits_sphere_S1_h2, hammerhits_p);
response_sphere_S1_h2( ~any(response_sphere_S1_h2,2), : ) = [];  %rows
hammerhits_sphere_S1_h2( ~any(hammerhits_sphere_S1_h2,2), : ) = [];  %rows
end


response_sphere_S1_h3= []; hammerhits_sphere_S1_h3= [];
for i=1:5 
I_response = data_sphere_S1_h3(2*i-1,:);
I_hammer= data_sphere_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_sphere_S1_h3= cat(1, response_sphere_S1_h3, response_p);
hammerhits_sphere_S1_h3= cat(1, hammerhits_sphere_S1_h3, hammerhits_p);
response_sphere_S1_h3( ~any(response_sphere_S1_h3,2), : ) = [];  %rows
hammerhits_sphere_S1_h3( ~any(hammerhits_sphere_S1_h3,2), : ) = [];  %rows
end


response_sphere_S1_h5= []; hammerhits_sphere_S1_h5= [];
for i=1:5 
I_response = data_sphere_S1_h5(2*i-1,:);
I_hammer= data_sphere_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_sphere_S1_h5= cat(1, response_sphere_S1_h5, response_p);
hammerhits_sphere_S1_h5= cat(1, hammerhits_sphere_S1_h5, hammerhits_p);
response_sphere_S1_h5( ~any(response_sphere_S1_h5,2), : ) = [];  %rows
hammerhits_sphere_S1_h5( ~any(hammerhits_sphere_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Cylinder hd %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_cylinder_S1_h2= []; hammerhits_cylinder_S1_h2= [];
for i=1:5 
I_response = data_cylinder_S1_h2(2*i-1,:);
I_hammer= data_cylinder_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_cylinder_S1_h2= cat(1, response_cylinder_S1_h2, response_p);
hammerhits_cylinder_S1_h2= cat(1, hammerhits_cylinder_S1_h2, hammerhits_p);
response_cylinder_S1_h2( ~any(response_cylinder_S1_h2,2), : ) = [];  %rows
hammerhits_cylinder_S1_h2( ~any(hammerhits_cylinder_S1_h2,2), : ) = [];  %rows
end

response_cylinder_S1_h3= []; hammerhits_cylinder_S1_h3= [];
for i=1:5 
I_response = data_cylinder_S1_h3(2*i-1,:);
I_hammer= data_cylinder_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_cylinder_S1_h3= cat(1, response_cylinder_S1_h3, response_p);
hammerhits_cylinder_S1_h3= cat(1, hammerhits_cylinder_S1_h3, hammerhits_p);
response_cylinder_S1_h3( ~any(response_cylinder_S1_h3,2), : ) = [];  %rows
hammerhits_cylinder_S1_h3( ~any(hammerhits_cylinder_S1_h3,2), : ) = [];  %rows
end

response_cylinder_S1_h5= []; hammerhits_cylinder_S1_h5= [];
for i=1:5 
I_response = data_cylinder_S1_h5(2*i-1,:);
I_hammer= data_cylinder_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_cylinder_S1_h5= cat(1, response_cylinder_S1_h5, response_p);
hammerhits_cylinder_S1_h5= cat(1, hammerhits_cylinder_S1_h5, hammerhits_p);
response_cylinder_S1_h5( ~any(response_cylinder_S1_h5,2), : ) = [];  %rows
hammerhits_cylinder_S1_h5( ~any(hammerhits_cylinder_S1_h5,2), : ) = [];  %rows
end


%%%%%%%%%%%%%%%%%%%%%%% Get responses for Sphere ld %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_sphereld_S1_h2= []; hammerhits_sphereld_S1_h2= [];
for i=1:5 
I_response = data_sphereld_S1_h2(2*i-1,:);
I_hammer= data_sphereld_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_sphereld_S1_h2= cat(1, response_sphereld_S1_h2, response_p);
hammerhits_sphereld_S1_h2= cat(1, hammerhits_sphereld_S1_h2, hammerhits_p);
response_sphereld_S1_h2( ~any(response_sphereld_S1_h2,2), : ) = [];  %rows
hammerhits_sphereld_S1_h2( ~any(hammerhits_sphereld_S1_h2,2), : ) = [];  %rows
end

response_sphereld_S1_h3= []; hammerhits_sphereld_S1_h3= [];
for i=1:5 
I_response = data_sphereld_S1_h3(2*i-1,:);
I_hammer= data_sphereld_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_sphereld_S1_h3= cat(1, response_sphereld_S1_h3, response_p);
hammerhits_sphereld_S1_h3= cat(1, hammerhits_sphereld_S1_h3, hammerhits_p);
response_sphereld_S1_h3( ~any(response_sphereld_S1_h3,2), : ) = [];  %rows
hammerhits_sphereld_S1_h3( ~any(hammerhits_sphereld_S1_h3,2), : ) = [];  %rows
end

response_sphereld_S1_h5= []; hammerhits_sphereld_S1_h5= [];
for i=1:5 
I_response = data_sphereld_S1_h5(2*i-1,:);
I_hammer= data_sphereld_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_sphereld_S1_h5= cat(1, response_sphereld_S1_h5, response_p);
hammerhits_sphereld_S1_h5= cat(1, hammerhits_sphereld_S1_h5, hammerhits_p);
response_sphereld_S1_h5( ~any(response_sphereld_S1_h5,2), : ) = [];  %rows
hammerhits_sphereld_S1_h5( ~any(hammerhits_sphereld_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Cylinder hd %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_cylinderld_S1_h2= []; hammerhits_cylinderld_S1_h2= [];
for i=1:5 
I_response = data_cylinderld_S1_h2(2*i-1,:);
I_hammer= data_cylinderld_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_cylinderld_S1_h2= cat(1, response_cylinderld_S1_h2, response_p);
hammerhits_cylinderld_S1_h2= cat(1, hammerhits_cylinderld_S1_h2, hammerhits_p);
response_cylinderld_S1_h2( ~any(response_cylinderld_S1_h2,2), : ) = [];  %rows
hammerhits_cylinderld_S1_h2( ~any(hammerhits_cylinderld_S1_h2,2), : ) = [];  %rows
end

response_cylinderld_S1_h3= []; hammerhits_cylinderld_S1_h3= [];
for i=1:5 
I_response = data_cylinderld_S1_h3(2*i-1,:);
I_hammer= data_cylinderld_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_cylinderld_S1_h3= cat(1, response_cylinderld_S1_h3, response_p);
hammerhits_cylinderld_S1_h3= cat(1, hammerhits_cylinderld_S1_h3, hammerhits_p);
response_cylinderld_S1_h3( ~any(response_cylinderld_S1_h3,2), : ) = [];  %rows
hammerhits_cylinderld_S1_h3( ~any(hammerhits_cylinderld_S1_h3,2), : ) = [];  %rows
end

response_cylinderld_S1_h5= []; hammerhits_cylinderld_S1_h5= [];
for i=1:5 
I_response = data_cylinderld_S1_h5(2*i-1,:);
I_hammer= data_cylinderld_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_cylinderld_S1_h5= cat(1, response_cylinderld_S1_h5, response_p);
hammerhits_cylinderld_S1_h5= cat(1, hammerhits_cylinderld_S1_h5, hammerhits_p);
response_cylinderld_S1_h5( ~any(response_cylinderld_S1_h5,2), : ) = [];  %rows
hammerhits_cylinderld_S1_h5( ~any(hammerhits_cylinderld_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for Circle %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_circle_S1_h2= []; hammerhits_circle_S1_h2= [];
for i=1:5 
I_response = data_circle_S1_h2(2*i-1,:);
I_hammer= data_circle_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_circle_S1_h2= cat(1, response_circle_S1_h2, response_p);
hammerhits_circle_S1_h2= cat(1, hammerhits_circle_S1_h2, hammerhits_p);
response_circle_S1_h2( ~any(response_circle_S1_h2,2), : ) = [];  %rows
hammerhits_circle_S1_h2( ~any(hammerhits_circle_S1_h2,2), : ) = [];  %rows
end

response_circle_S1_h3= []; hammerhits_circle_S1_h3= [];
for i=1:5 
I_response = data_circle_S1_h3(2*i-1,:);
I_hammer= data_circle_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_circle_S1_h3= cat(1, response_circle_S1_h3, response_p);
hammerhits_circle_S1_h3= cat(1, hammerhits_circle_S1_h3, hammerhits_p);
response_circle_S1_h3( ~any(response_circle_S1_h3,2), : ) = [];  %rows
hammerhits_circle_S1_h3( ~any(hammerhits_circle_S1_h3,2), : ) = [];  %rows
end

response_circle_S1_h5= []; hammerhits_circle_S1_h5= [];
for i=1:5 
I_response = data_circle_S1_h5(2*i-1,:);
I_hammer= data_circle_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_circle_S1_h5= cat(1, response_circle_S1_h5, response_p);
hammerhits_circle_S1_h5= cat(1, hammerhits_circle_S1_h5, hammerhits_p);
response_circle_S1_h5( ~any(response_circle_S1_h5,2), : ) = [];  %rows
hammerhits_circle_S1_h5( ~any(hammerhits_circle_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for slit %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_slit_S1_h2= []; hammerhits_slit_S1_h2= [];
for i=1:5 
I_response = data_slit_S1_h2(2*i-1,:);
I_hammer= data_slit_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_slit_S1_h2= cat(1, response_slit_S1_h2, response_p);
hammerhits_slit_S1_h2= cat(1, hammerhits_slit_S1_h2, hammerhits_p);
response_slit_S1_h2( ~any(response_slit_S1_h2,2), : ) = [];  %rows
hammerhits_slit_S1_h2( ~any(hammerhits_slit_S1_h2,2), : ) = [];  %rows
end

response_slit_S1_h3= []; hammerhits_slit_S1_h3= [];
for i=1:5 
I_response = data_slit_S1_h3(2*i-1,:);
I_hammer= data_slit_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_slit_S1_h3= cat(1, response_slit_S1_h3, response_p);
hammerhits_slit_S1_h3= cat(1, hammerhits_slit_S1_h3, hammerhits_p);
response_slit_S1_h3( ~any(response_slit_S1_h3,2), : ) = [];  %rows
hammerhits_slit_S1_h3( ~any(hammerhits_slit_S1_h3,2), : ) = [];  %rows
end

response_slit_S1_h5= []; hammerhits_slit_S1_h5= [];
for i=1:5 
I_response = data_slit_S1_h5(2*i-1,:);
I_hammer= data_slit_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_slit_S1_h5= cat(1, response_slit_S1_h5, response_p);
hammerhits_slit_S1_h5= cat(1, hammerhits_slit_S1_h5, hammerhits_p);
response_slit_S1_h5( ~any(response_slit_S1_h5,2), : ) = [];  %rows
hammerhits_slit_S1_h5( ~any(hammerhits_slit_S1_h5,2), : ) = [];  %rows
end


%%%%%%%%%%%%%%%%%%%%%%% Get responses for ssquare %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_ssquare_S1_h2= []; hammerhits_ssquare_S1_h2= [];
for i=1:5 
I_response = data_ssquare_S1_h2(2*i-1,:);
I_hammer= data_ssquare_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_ssquare_S1_h2= cat(1, response_ssquare_S1_h2, response_p);
hammerhits_ssquare_S1_h2= cat(1, hammerhits_ssquare_S1_h2, hammerhits_p);
response_ssquare_S1_h2( ~any(response_ssquare_S1_h2,2), : ) = [];  %rows
hammerhits_ssquare_S1_h2( ~any(hammerhits_ssquare_S1_h2,2), : ) = [];  %rows
end

response_ssquare_S1_h3= []; hammerhits_ssquare_S1_h3= [];
for i=1:5 
I_response = data_ssquare_S1_h3(2*i-1,:);
I_hammer= data_ssquare_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_ssquare_S1_h3= cat(1, response_ssquare_S1_h3, response_p);
hammerhits_ssquare_S1_h3= cat(1, hammerhits_ssquare_S1_h3, hammerhits_p);
response_ssquare_S1_h3( ~any(response_ssquare_S1_h3,2), : ) = [];  %rows
hammerhits_ssquare_S1_h3( ~any(hammerhits_ssquare_S1_h3,2), : ) = [];  %rows
end

response_ssquare_S1_h5= []; hammerhits_ssquare_S1_h5= [];
for i=1:5 
I_response = data_ssquare_S1_h5(2*i-1,:);
I_hammer= data_ssquare_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_ssquare_S1_h5= cat(1, response_ssquare_S1_h5, response_p);
hammerhits_ssquare_S1_h5= cat(1, hammerhits_ssquare_S1_h5, hammerhits_p);
response_ssquare_S1_h5( ~any(response_ssquare_S1_h5,2), : ) = [];  %rows
hammerhits_ssquare_S1_h5( ~any(hammerhits_ssquare_S1_h5,2), : ) = [];  %rows
end

%%%%%%%%%%%%%%%%%%%%%%% Get responses for bsquare %%%%%%%%%%%%%%%%%%%%%%%%%%

% Location S1
response_bsquare_S1_h2= []; hammerhits_bsquare_S1_h2= [];
for i=1:5 
I_response = data_bsquare_S1_h2(2*i-1,:);
I_hammer= data_bsquare_S1_h2(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_bsquare_S1_h2= cat(1, response_bsquare_S1_h2, response_p);
hammerhits_bsquare_S1_h2= cat(1, hammerhits_bsquare_S1_h2, hammerhits_p);
response_bsquare_S1_h2( ~any(response_bsquare_S1_h2,2), : ) = [];  %rows
hammerhits_bsquare_S1_h2( ~any(hammerhits_bsquare_S1_h2,2), : ) = [];  %rows
end

response_bsquare_S1_h3= []; hammerhits_bsquare_S1_h3= [];
for i=1:5 
I_response = data_bsquare_S1_h3(2*i-1,:);
I_hammer= data_bsquare_S1_h3(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_bsquare_S1_h3= cat(1, response_bsquare_S1_h3, response_p);
hammerhits_bsquare_S1_h3= cat(1, hammerhits_bsquare_S1_h3, hammerhits_p);
response_bsquare_S1_h3( ~any(response_bsquare_S1_h3,2), : ) = [];  %rows
hammerhits_bsquare_S1_h3( ~any(hammerhits_bsquare_S1_h3,2), : ) = [];  %rows
end

response_bsquare_S1_h5= []; hammerhits_bsquare_S1_h5= [];
for i=1:5 
I_response = data_bsquare_S1_h5(2*i-1,:);
I_hammer= data_bsquare_S1_h5(2*i,:); 
[response_p, hammerhits_p] = findpeaks_c(I_response, I_hammer, Tlen); 

response_bsquare_S1_h5= cat(1, response_bsquare_S1_h5, response_p);
hammerhits_bsquare_S1_h5= cat(1, hammerhits_bsquare_S1_h5, hammerhits_p);
response_bsquare_S1_h5( ~any(response_bsquare_S1_h5,2), : ) = [];  %rows
hammerhits_bsquare_S1_h5( ~any(hammerhits_bsquare_S1_h5,2), : ) = [];  %rows
end

clear response_p hammerhits_p

%% 3. Find the maximum hammer strength of all the recordings 
% To account for variations in the impulse strength applied on the
% specimens we need to first find the maximum impulse strength applied to
% all the specimens. Depending on this max strength we will scale up the
% responses of the other subsequences so that all the subseqeunces would
% effectively have the same impulse applied to the specimen. 

userValue = input('Want scaled features? [1=yes 0=no]: ');


% Find impulse strength for each subsequence
vars = who;
hammerVars = vars(startsWith(vars, 'hammerhits'));

% Initialize an empty array to store max values
allMaxVals_H = [];

% Loop over each 'hammerhits' variable
for i = 1:length(hammerVars)
    data = eval(hammerVars{i}); 
    rowMax = max(data, [], 2);  
    allMaxVals_H = [allMaxVals_H; rowMax];
end

max_hammer_strength = max(allMaxVals_H);
[maxValue_H, maxIndex_H] = max(allMaxVals_H); % Find the hardest tap

% Loop over each 'response' variable to find max response
responseVars = vars(startsWith(vars, 'response'));
allMaxVals_R = [];
for i = 1:length(responseVars)
    data = eval(responseVars{i}); 
    rowMax = max(abs(data), [], 2);  
    allMaxVals_R = [allMaxVals_R; rowMax];
end

max_response_amplitude = max(allMaxVals_R);

% Visualize all peaks and amplitudes    
figure();
scatter(allMaxVals_H, allMaxVals_R, 'filled');
p = polyfit(allMaxVals_H, allMaxVals_R, 1);       
yfit = polyval(p, allMaxVals_H);  

% Compute R-squared
yresid = allMaxVals_R - yfit;                  
SSresid = sum(yresid.^2);                   
SStotal = sum((allMaxVals_R - mean(allMaxVals_R)).^2);  
Rsq = 1 - SSresid/SStotal;                    
% Annotate with R²
xpos = min(allMaxVals_H) + 0.05 * range(allMaxVals_H); % 5% from left
ypos = max(allMaxVals_R) - 0.05 * range(allMaxVals_R); % 5% from top
text(xpos, ypos, sprintf('R^2 = %.4f', Rsq), ...
    'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');

hold on;
plot(allMaxVals_H, yfit, 'r-', 'LineWidth', 2);    % Plot the fitted line
xlabel('Impulse strength [V]');
ylabel('Response amplitude [acceleration, m/s^2]');
grid on;

%% 4. Apply Scaling to each specimen according to hammer impulse

if userValue==1

    disp('Running scaling case...');
    % Normal0 
    max_values = max(abs(hammerhits_normal0_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal0_S1_h2 = response_normal0_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal0_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal0_S1_h3 = response_normal0_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal0_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal0_S1_h5 = response_normal0_S1_h5./scalar_param;
    
    % Normal1 
    max_values = max(abs(hammerhits_normal1_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal1_S1_h2 = response_normal1_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal1_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal1_S1_h3 = response_normal1_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal1_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal1_S1_h5 = response_normal1_S1_h5./scalar_param;

    % Normal2 
    max_values = max(abs(hammerhits_normal2_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal2_S1_h2 = response_normal2_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal2_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal2_S1_h3 = response_normal2_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal2_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal2_S1_h5 = response_normal2_S1_h5./scalar_param;

    % Normal3 
    max_values = max(abs(hammerhits_normal3_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal3_S1_h2 = response_normal3_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal3_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal3_S1_h3 = response_normal3_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal3_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal3_S1_h5 = response_normal3_S1_h5./scalar_param;

    % Normal4 
    max_values = max(abs(hammerhits_normal4_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal4_S1_h2 = response_normal4_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal4_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal4_S1_h3 = response_normal4_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal4_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal4_S1_h5 = response_normal4_S1_h5./scalar_param;

    % Normal5 
    max_values = max(abs(hammerhits_normal5_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal5_S1_h2 = response_normal5_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal5_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal5_S1_h3 = response_normal5_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal5_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal5_S1_h5 = response_normal5_S1_h5./scalar_param;

    % Normal6 
    max_values = max(abs(hammerhits_normal6_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal6_S1_h2 = response_normal6_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal6_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal6_S1_h3 = response_normal6_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal6_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal6_S1_h5 = response_normal6_S1_h5./scalar_param;

    % Normal7 
    max_values = max(abs(hammerhits_normal7_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal7_S1_h2 = response_normal7_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal7_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal7_S1_h3 = response_normal7_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal7_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal7_S1_h5 = response_normal7_S1_h5./scalar_param;

    % Normal8 
    max_values = max(abs(hammerhits_normal8_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal8_S1_h2 = response_normal8_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal8_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal8_S1_h3 = response_normal8_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal8_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal8_S1_h5 = response_normal8_S1_h5./scalar_param;

    % Normal9 
    max_values = max(abs(hammerhits_normal9_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal9_S1_h2 = response_normal9_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal9_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal9_S1_h3 = response_normal9_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal9_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal9_S1_h5 = response_normal9_S1_h5./scalar_param;

    % Normal10 
    max_values = max(abs(hammerhits_normal10_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal10_S1_h2 = response_normal10_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_normal10_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal10_S1_h3 = response_normal10_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_normal10_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_normal10_S1_h5 = response_normal10_S1_h5./scalar_param;

    % Sphere hd 
    max_values = max(abs(hammerhits_sphere_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_sphere_S1_h2 = response_sphere_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_sphere_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_sphere_S1_h3 = response_sphere_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_sphere_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_sphere_S1_h5 = response_sphere_S1_h5./scalar_param;

    % Cylinder hd 
    max_values = max(abs(hammerhits_cylinder_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_cylinder_S1_h2 = response_cylinder_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_cylinder_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_cylinder_S1_h3 = response_cylinder_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_cylinder_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_cylinder_S1_h5 = response_cylinder_S1_h5./scalar_param;

    % Sphere ld
    max_values = max(abs(hammerhits_sphereld_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_sphereld_S1_h2 = response_sphereld_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_sphereld_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_sphereld_S1_h3 = response_sphereld_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_sphereld_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_sphereld_S1_h5 = response_sphereld_S1_h5./scalar_param;

    % Cylinder ld
    max_values = max(abs(hammerhits_cylinderld_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_cylinderld_S1_h2 = response_cylinderld_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_cylinderld_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_cylinderld_S1_h3 = response_cylinderld_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_cylinderld_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_cylinderld_S1_h5 = response_cylinderld_S1_h5./scalar_param;

    % Circle
    max_values = max(abs(hammerhits_circle_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_circle_S1_h2 = response_circle_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_circle_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_circle_S1_h3 = response_circle_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_circle_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_circle_S1_h5 = response_circle_S1_h5./scalar_param;

    % Slit
    max_values = max(abs(hammerhits_slit_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_slit_S1_h2 = response_slit_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_slit_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_slit_S1_h3 = response_slit_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_slit_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_slit_S1_h5 = response_slit_S1_h5./scalar_param;

    % Ssquare
    max_values = max(abs(hammerhits_ssquare_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_ssquare_S1_h2 = response_ssquare_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_ssquare_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_ssquare_S1_h3 = response_ssquare_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_ssquare_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_ssquare_S1_h5 = response_ssquare_S1_h5./scalar_param;

    % Bsquare
    max_values = max(abs(hammerhits_bsquare_S1_h2), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_bsquare_S1_h2 = response_bsquare_S1_h2./scalar_param;
    
    max_values = max(abs(hammerhits_bsquare_S1_h3), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_bsquare_S1_h3 = response_bsquare_S1_h3./scalar_param;
    
    max_values = max(abs(hammerhits_bsquare_S1_h5), [], 2);
    scalar_param = max_values ./ maxValue_H;
    response_bsquare_S1_h5 = response_bsquare_S1_h5./scalar_param;

end

%% 5. Normalize the amplitudes of the response to be from 0 to 1
if userValue==1

    % Loop over each 'response' variable to find max response
    responseVars = vars(startsWith(vars, 'response'));
    allMaxVals_R = [];
    for i = 1:length(responseVars)
        data = eval(responseVars{i}); 
        rowMax = max(abs(data), [], 2);  
        allMaxVals_R = [allMaxVals_R; rowMax];
    end
    
    max_response_amplitude = max(allMaxVals_R);
    
    % Now perform normalization!

    % Normal0 
    response_normal0_S1_h2 = response_normal0_S1_h2./max_response_amplitude;
    response_normal0_S1_h3 = response_normal0_S1_h3./max_response_amplitude;
    response_normal0_S1_h5 = response_normal0_S1_h5./max_response_amplitude;
    
    % Normal1 
    response_normal1_S1_h2 = response_normal1_S1_h2./max_response_amplitude;
    response_normal1_S1_h3 = response_normal1_S1_h3./max_response_amplitude;
    response_normal1_S1_h5 = response_normal1_S1_h5./max_response_amplitude;

    % Normal2 
    response_normal2_S1_h2 = response_normal2_S1_h2./max_response_amplitude;
    response_normal2_S1_h3 = response_normal2_S1_h3./max_response_amplitude;
    response_normal2_S1_h5 = response_normal2_S1_h5./max_response_amplitude;

    % Normal3 
    response_normal3_S1_h2 = response_normal3_S1_h2./max_response_amplitude;
    response_normal3_S1_h3 = response_normal3_S1_h3./max_response_amplitude;
    response_normal3_S1_h5 = response_normal3_S1_h5./max_response_amplitude;

    % Normal4 
    response_normal4_S1_h2 = response_normal4_S1_h2./max_response_amplitude;
    response_normal4_S1_h3 = response_normal4_S1_h3./max_response_amplitude;
    response_normal4_S1_h5 = response_normal4_S1_h5./max_response_amplitude;

    % Normal5 
    response_normal5_S1_h2 = response_normal5_S1_h2./max_response_amplitude;
    response_normal5_S1_h3 = response_normal5_S1_h3./max_response_amplitude;
    response_normal5_S1_h5 = response_normal5_S1_h5./max_response_amplitude;

    % Normal6 
    response_normal6_S1_h2 = response_normal6_S1_h2./max_response_amplitude;
    response_normal6_S1_h3 = response_normal6_S1_h3./max_response_amplitude;
    response_normal6_S1_h5 = response_normal6_S1_h5./max_response_amplitude;

    % Normal7 
    response_normal7_S1_h2 = response_normal7_S1_h2./max_response_amplitude;
    response_normal7_S1_h3 = response_normal7_S1_h3./max_response_amplitude;
    response_normal7_S1_h5 = response_normal7_S1_h5./max_response_amplitude;

    % Normal8 
    response_normal8_S1_h2 = response_normal8_S1_h2./max_response_amplitude;
    response_normal8_S1_h3 = response_normal8_S1_h3./max_response_amplitude;
    response_normal8_S1_h5 = response_normal8_S1_h5./max_response_amplitude;

    % Normal9 
    response_normal9_S1_h2 = response_normal9_S1_h2./max_response_amplitude;
    response_normal9_S1_h3 = response_normal9_S1_h3./max_response_amplitude;
    response_normal9_S1_h5 = response_normal9_S1_h5./max_response_amplitude;

    % Normal10 
    response_normal10_S1_h2 = response_normal10_S1_h2./max_response_amplitude;
    response_normal10_S1_h3 = response_normal10_S1_h3./max_response_amplitude;
    response_normal10_S1_h5 = response_normal10_S1_h5./max_response_amplitude;

    % Sphere hd 
    response_sphere_S1_h2 = response_sphere_S1_h2./max_response_amplitude;
    response_sphere_S1_h3 = response_sphere_S1_h3./max_response_amplitude;
    response_sphere_S1_h5 = response_sphere_S1_h5./max_response_amplitude;

    % Cylinder hd 
    response_cylinder_S1_h2 = response_cylinder_S1_h2./max_response_amplitude;
    response_cylinder_S1_h3 = response_cylinder_S1_h3./max_response_amplitude;
    response_cylinder_S1_h5 = response_cylinder_S1_h5./max_response_amplitude;

    % Sphere ld
    response_sphereld_S1_h2 = response_sphereld_S1_h2./max_response_amplitude;
    response_sphereld_S1_h3 = response_sphereld_S1_h3./max_response_amplitude;
    response_sphereld_S1_h5 = response_sphereld_S1_h5./max_response_amplitude;

    % Cylinder ld
    response_cylinderld_S1_h2 = response_cylinderld_S1_h2./max_response_amplitude;
    response_cylinderld_S1_h3 = response_cylinderld_S1_h3./max_response_amplitude;
    response_cylinderld_S1_h5 = response_cylinderld_S1_h5./max_response_amplitude;

    % Circle
    response_circle_S1_h2 = response_circle_S1_h2./max_response_amplitude;
    response_circle_S1_h3 = response_circle_S1_h3./max_response_amplitude;
    response_circle_S1_h5 = response_circle_S1_h5./max_response_amplitude;

    % Slit
    response_slit_S1_h2 = response_slit_S1_h2./max_response_amplitude;
    response_slit_S1_h3 = response_slit_S1_h3./max_response_amplitude;
    response_slit_S1_h5 = response_slit_S1_h5./max_response_amplitude;

    % Ssquare
    response_ssquare_S1_h2 = response_ssquare_S1_h2./max_response_amplitude;
    response_ssquare_S1_h3 = response_ssquare_S1_h3./max_response_amplitude;
    response_ssquare_S1_h5 = response_ssquare_S1_h5./max_response_amplitude;

    % Bsquare
    response_bsquare_S1_h2 = response_bsquare_S1_h2./max_response_amplitude;
    response_bsquare_S1_h3 = response_bsquare_S1_h3./max_response_amplitude;
    response_bsquare_S1_h5 = response_bsquare_S1_h5./max_response_amplitude;
end
%% 6. Saving the isolated subsequences for analysis on associated python file
% Now save the extracted subsequences for dimensionality reduction on the
% second python file

% Normal specimens
save('features_normal0_S1_H2.mat', 'response_normal0_S1_h2')
save('features_normal0_S1_H3.mat', 'response_normal0_S1_h3')
save('features_normal0_S1_H5.mat', 'response_normal0_S1_h5')

save('features_normal1_S1_H2.mat', 'response_normal1_S1_h2')
save('features_normal1_S1_H3.mat', 'response_normal1_S1_h3')
save('features_normal1_S1_H5.mat', 'response_normal1_S1_h5')

save('features_normal2_S1_H2.mat', 'response_normal2_S1_h2')
save('features_normal2_S1_H3.mat', 'response_normal2_S1_h3')
save('features_normal2_S1_H5.mat', 'response_normal2_S1_h5')

save('features_normal3_S1_H2.mat', 'response_normal3_S1_h2')
save('features_normal3_S1_H3.mat', 'response_normal3_S1_h3')
save('features_normal3_S1_H5.mat', 'response_normal3_S1_h5')

save('features_normal4_S1_H2.mat', 'response_normal4_S1_h2')
save('features_normal4_S1_H3.mat', 'response_normal4_S1_h3')
save('features_normal4_S1_H5.mat', 'response_normal4_S1_h5')

save('features_normal5_S1_H2.mat', 'response_normal5_S1_h2')
save('features_normal5_S1_H3.mat', 'response_normal5_S1_h3')
save('features_normal5_S1_H5.mat', 'response_normal5_S1_h5')

save('features_normal6_S1_H2.mat', 'response_normal6_S1_h2')
save('features_normal6_S1_H3.mat', 'response_normal6_S1_h3')
save('features_normal6_S1_H5.mat', 'response_normal6_S1_h5')

save('features_normal7_S1_H2.mat', 'response_normal7_S1_h2')
save('features_normal7_S1_H3.mat', 'response_normal7_S1_h3')
save('features_normal7_S1_H5.mat', 'response_normal7_S1_h5')

save('features_normal8_S1_H2.mat', 'response_normal8_S1_h2')
save('features_normal8_S1_H3.mat', 'response_normal8_S1_h3')
save('features_normal8_S1_H5.mat', 'response_normal8_S1_h5')

save('features_normal9_S1_H2.mat', 'response_normal9_S1_h2')
save('features_normal9_S1_H3.mat', 'response_normal9_S1_h3')
save('features_normal9_S1_H5.mat', 'response_normal9_S1_h5')

save('features_normal10_S1_H2.mat', 'response_normal10_S1_h2')
save('features_normal10_S1_H3.mat', 'response_normal10_S1_h3')
save('features_normal10_S1_H5.mat', 'response_normal10_S1_h5')

% Defect specimens

save('features_spherehd_S1_H2.mat', 'response_sphere_S1_h2')
save('features_spherehd_S1_H3.mat', 'response_sphere_S1_h3')
save('features_spherehd_S1_H5.mat', 'response_sphere_S1_h5')

save('features_cylinderhd_S1_H2.mat', 'response_cylinder_S1_h2')
save('features_cylinderhd_S1_H3.mat', 'response_cylinder_S1_h3')
save('features_cylinderhd_S1_H5.mat', 'response_cylinder_S1_h5')

save('features_circle_S1_H2.mat', 'response_circle_S1_h2')
save('features_circle_S1_H3.mat', 'response_circle_S1_h3')
save('features_circle_S1_H5.mat', 'response_circle_S1_h5')

save('features_sphereld_S1_H2.mat', 'response_sphereld_S1_h2')
save('features_sphereld_S1_H3.mat', 'response_sphereld_S1_h3')
save('features_sphereld_S1_H5.mat', 'response_sphereld_S1_h5')

save('features_cylinderld_S1_H2.mat', 'response_cylinderld_S1_h2')
save('features_cylinderld_S1_H3.mat', 'response_cylinderld_S1_h3')
save('features_cylinderld_S1_H5.mat', 'response_cylinderld_S1_h5')

save('features_slit_S1_H2.mat', 'response_slit_S1_h2')
save('features_slit_S1_H3.mat', 'response_slit_S1_h3')
save('features_slit_S1_H5.mat', 'response_slit_S1_h5')

save('features_bsquare_S1_H2.mat', 'response_bsquare_S1_h2')
save('features_bsquare_S1_H3.mat', 'response_bsquare_S1_h3')
save('features_bsquare_S1_H5.mat', 'response_bsquare_S1_h5')

save('features_ssquare_S1_H2.mat', 'response_ssquare_S1_h2')
save('features_ssquare_S1_H3.mat', 'response_ssquare_S1_h3')
save('features_ssquare_S1_H5.mat', 'response_ssquare_S1_h5')

disp('Done running!')