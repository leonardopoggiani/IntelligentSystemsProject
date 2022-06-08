clear
close all
clc
format compact

%% Preparation of Data

% Dataset load
dataset = load('../datasets/dataset.mat'); 
dataset = table2array(dataset.dataset);
[howManyRows, ~] = size(dataset);

% Removal of non numerical values
inf_val = isinf(dataset);
[rows_inf, col_inf] = find(inf_val == 1);
dataset(rows_inf,:) = [];

[non_numerical_rows, ~] = size(dataset);
ret = howManyRows - non_numerical_rows;
fprintf("%i non numerical values removed\n", ret);

% Removal of outliers
dataset = dataset(:, 3:end);
dataset = rmoutliers(dataset);

X = dataset(:,3:end);
t_arousal = dataset(:,1);
t_valence = dataset(:,2);

[final_rows, ~] = size(dataset);

ret = non_numerical_rows - final_rows;
fprintf("%i outliers removed\n", ret);


%% Data Balancing
arousal_level = dataset(:,1);
valence_level = dataset(:,2);

% samples for arousal and valence
sample_arousal = groupcounts(arousal_level);
sample_valence = groupcounts(valence_level);

% plot the graph
figure("Name", "Sample for arousal before balancing");
bar(sample_arousal);
title("Sample for arousal before balancing");

fprintf("Data are unbalanced\n");