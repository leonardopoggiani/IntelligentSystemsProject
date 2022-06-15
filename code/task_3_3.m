%% Clean
clear
close all
clc
format compact

%% Load dataset

best3 = load("data/best3.mat");
x_train = best3.best3.x_train;
y_train = best3.best3.y_train;
x_test = best3.best3.x_test;
y_test = best3.best3.y_test;
best_features = best3.best3.best_features;
y_values = best3.best3.y_values;

%Retrive the features name from the entire dataset
dataset = load("data/dataset.mat");
features_names = dataset.dataset.Properties.VariableNames(5:58);
best3_features_names = features_names(best_features);

x_27= x_train(:,1);
x_11=x_train(:,2);
x_13=x_train(:,3);



%Plot histogram
figure(1)
t = tiledlayout(1,3);
nexttile
histogram(x_27,'BinWidth',0.5);
title('GSR1');
nexttile
histogram(x_11, 'BinWidth',0.5);
x_11_name = best3_features_names(:,2);
title('ECG28');
nexttile
histogram(x_13, 'BinWidth',0.5);
x_13_name = best3_features_names(:,3);
title('ECG30');

max_27 = max(x_27);
max_11 = max(x_11);
max_13 = max(x_13);

min_27 = min(x_27);
min_11 = min(x_11);
min_13 = min(x_13);

fprintf(" --- RANGES FOR UNIVERSE OF DISCOURSE ---\n");
fprintf("  Feature 27 -> Max:%f Min:%f\n", max_27, min_27);
fprintf("  Feature 11 -> Max:%f Min:%f\n", max_11, min_11);
fprintf("  Feature 13 -> Max:%f Min:%f\n", max_13, min_13);


%% Index of samples for each output
index1 = find(y_train == y_values(1));
index2 = find(y_train == y_values(2)); 
index3 = find(y_train == y_values(3));
index4 = find(y_train == y_values(4));
index5 = find(y_train == y_values(5));
index6 = find(y_train == y_values(6));
index7 = find(y_train == y_values(7));

low = [index1 index2];
middle = [index3 index4 index5];
high = [index6 index7];

% Plot Histogram of features for a specific subset of outputs
binwidth = 0.5;
y_lim = 30;

figure(2)
t = tiledlayout(1,3);
nexttile

%Feature 27
histogram(x_27(low), 'BinWidth', binwidth);
yline(y_lim, '--r');
title('GRS1 low');
nexttile
histogram(x_27(middle), 'BinWidth', 0.5);
yline(y_lim, '--r');
title('GRS1 medium');
nexttile
histogram(x_27(high), 'BinWidth', binwidth);
yline(y_lim, '--r');
title('GRS1 high');


%Feature 11
figure(3)
t = tiledlayout(1,3);
nexttile
histogram(x_11(low), 'BinWidth', binwidth);
yline(y_lim, '--r');
title('ECG28 low');
nexttile
histogram(x_11(middle), 'BinWidth', 0.5);
yline(y_lim, '--r');
title('ECG28 medium');
nexttile
histogram(x_11(high), 'BinWidth', binwidth);
yline(y_lim, '--r');
title('ECG28 high');

%Feature 13
figure(4)
t = tiledlayout(1,3);
nexttile
histogram(x_13(low), 'BinWidth', binwidth);
yline(y_lim, '--r');
title('ECG30 low');
nexttile
histogram(x_13(middle), 'BinWidth', binwidth);
yline(y_lim, '--r');
title('ECG30 medium');
nexttile
histogram(x_13(high), 'BinWidth', binwidth);
yline(y_lim, '--r');
title('ECG30 high');

%% Plot Scatterplots between pairs of features to find some correlations

figure(5)
t = tiledlayout(1,3);
scatter(x_27, x_11);
title('Scatterplot of feature 27 and feature 11');
figure(6)
scatter(x_27, x_13);
title('Scatterplot of feature 27 and feature 13');
figure(7)
scatter(x_11, x_13);
title('Scatterplot of feature 11 and feature 13');


%% Evaluation of Mamdani
fis = readfis('Mamdani');
output=evalfis(fis, [1 1 2]);