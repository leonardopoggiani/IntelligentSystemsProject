%% Clean
clear
close all
clc
format compact

%% Load the features

test_arousal = load('data/testing_arousal.mat');
train_arousal = load('data/training_arousal.mat');
x_train_arousal = train_arousal.best_arousal_training.x_train';
y_train_arousal = train_arousal.best_arousal_training.y_train'.';
x_test_arousal = test_arousal.best_arousal_testing.x_test';
y_test_arousal = test_arousal.best_arousal_testing.y_test'.';

fprintf("Arousal features loaded\n");


test_valence = load('data/testing_valence.mat');
train_valence = load('data/training_valence.mat');
x_train_valence = train_valence.best_valance_training.x_train';
y_train_valence = train_valence.best_valance_training.y_train'.';
x_test_valence = test_valence.best_valance_testing.x_test';
y_test_valence = test_valence.best_valance_testing.y_test'.';

fprintf("Valence features loaded\n");

MLP_AROUSAL = 0;
MLP_VALENCE = 0;
RBFN_AROUSAL = 0;
RBFN_VALENCE = 1;

%% MLP for Arousal

if MLP_AROUSAL == 1
    % Creation of MLP
    hiddenLayerSize_arousal = 30;
    mlp_arousal = fitnet(hiddenLayerSize_arousal);
    mlp_arousal.divideParam.trainRatio = 0.7;
    mlp_arousal.divideParam.testRatio = 0.1;
    mlp_arousal.divideParam.valRatio = 0.2;
    mlp_arousal.trainParam.showCommandLine=1;
    %mlp_arousal.trainParam.showWindow=0;
    mlp_arousal.trainParam.epochs =110;
    %Training
    [mlp_arousal, tr] = train(mlp_arousal, x_train_arousal, y_train_arousal);
    view(mlp_arousal);
    figure(1);
    plotperform(tr);
    % Test
    test_output_arousal = mlp_arousal(x_test_arousal);
    % Plot regression
    figure(2)
    plotregression(y_test_arousal, test_output_arousal, " Arousal ");
end

%% MLP for Valence

if MLP_VALENCE == 1
    %Creation of MLP
    hiddenLayerSize_valence = 35;
    mlp_valence = fitnet(hiddenLayerSize_valence);
    mlp_valence.divideParam.trainRatio = 0.7;
    mlp_valence.divideParam.testRatio = 0.1;
    mlp_valence.divideParam.valRatio = 0.2;
    mlp_valence.trainParam.showCommandLine=1;
    %mlp_valence.trainParam.showWindow=0;
    mlp_valence.trainParam.epochs =110;
    %Training
    [mlp_valence, tr_v] = train(mlp_valence, x_train_valence, y_train_valence);
    view(mlp_valence);
    figure(3);
    plotperform(tr_v);
    % Test
    test_output_valence = mlp_valence(x_test_valence);
    % Plot regression
    figure(4)
    plotregression(y_test_valence, test_output_valence, " Valence ");
end

%% RBFN for Arousal
if RBFN_AROUSAL == 1
    %Creation of RBFN
    goal_ar = 0;
    spread_ar = 1;
    K_ar = 400;
    rbf_arousal = newrb(x_train_arousal, y_train_arousal, goal_ar, spread_ar, K_ar);
    view (rbf_arousal);
    %Test
    test_output_arousal_rbf = rbf_arousal(x_test_arousal);
    figure(5)
    %Plot regression
    plotregression(y_test_arousal, test_output_arousal_rbf, 'Arousal:');
end

%% RBFN for Valence

if RBFN_VALENCE == 1
    %Creation of RBFN
    goal_vl = 0;
    spread_vl = 1;
    K_vl = 350;
    rbf_valence = newrb(x_train_valence, y_train_valence, goal_vl, spread_vl, K_vl);
    view (rbf_valence);
    %Test
    test_output_valence_rbf = rbf_valence(x_test_valence);
    %Plot regression
    figure(6)
    plotregression(y_test_valence, test_output_valence_rbf, 'Valence:');
end