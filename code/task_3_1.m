%% Preparation of data
clear
close all
clc

%Load datasets
dataset = load('../datasets/dataset.mat'); 
dataset = table2array(dataset.dataset);

%Load results obtained from sequential feature selection
data_prep = load('data_preparation_results_100.mat');

%Data for training
X_train_arousal = data_prep.X_train_best_arousal;
X_train_valence = data_prep.X_train_best_valence;
t_train_arousal = data_prep.t_train_best_arousal;
t_train_valence = data_prep.t_train_best_valence;

%Data for final test of the network to assess performance
X_test_arousal = data_prep.X_test_best_arousal;
X_test_valence = data_prep.X_test_best_valence;
t_test_arousal = data_prep.t_test_best_arousal;
t_test_valence = data_prep.t_test_best_valence;

%% Training MLP FOR AROUSAL


% Optimal Neural Network Architecture found for arousal
mlp_net_arousal = fitnet(45);
mlp_net_arousal.divideParam.trainRatio = 0.7;
mlp_net_arousal.divideParam.testRatio = 0.1; %Just to see the difference with the other test set
mlp_net_arousal.divideParam.valRatio = 0.2;
mlp_net_arousal.trainParam.showWindow = 1;
mlp_net_arousal.trainParam.showCommandLine = 1;
mlp_net_arousal.trainParam.lr = 0.1; 
mlp_net_arousal.trainParam.epochs = 100;
mlp_net_arousal.trainParam.max_fail = 10;

[mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, X_train_arousal, t_train_arousal);
view(mlp_net_arousal);
y_test_arousal = mlp_net_arousal(X_test_arousal);
plotregression(t_test_arousal, y_test_arousal, ['Final test arousal 45 neurons: ']);


% Traces of other experiments
% max_neurons_1 = 120;
% for i=5:5:max_neurons_1    
%     mlp_net_arousal = fitnet(i);
%     mlp_net_arousal.divideParam.trainRatio = 0.7;
%     mlp_net_arousal.divideParam.testRatio = 0.1; 
%     mlp_net_arousal.divideParam.valRatio = 0.2;
%     mlp_net_arousal.trainParam.showWindow = 0;
%     mlp_net_arousal.trainParam.showCommandLine = 1;
%     mlp_net_arousal.trainParam.lr = 0.05; 
%     mlp_net_arousal.trainParam.epochs = 100;
%     mlp_net_arousal.trainParam.max_fail = 10;
%     [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, X_train_arousal, t_train_arousal);
%     
%     y_test_arousal = mlp_net_arousal(X_test_arousal);
%     plotregression(t_test_arousal, y_test_arousal, ['Final test arousal: ' string(i)]);
% end

%%  train mlp for valence


% Optimal Neural Network Architecture found for valence
mlp_net_valence = fitnet(80);
mlp_net_valence.divideParam.trainRatio = 0.7; 
mlp_net_valence.divideParam.valRatio = 0.2; 
mlp_net_valence.divideParam.testRatio = 0.1;
mlp_net_valence.trainParam.showWindow = 1;
mlp_net_valence.trainParam.showCommandLine = 1;
mlp_net_valence.trainParam.lr = 0.1; 
mlp_net_valence.trainParam.epochs = 100;
mlp_net_valence.trainParam.max_fail = 15;
[mlp_net_valence, tr_valence] = train(mlp_net_valence, X_train_valence, t_train_valence);
y_test_valence = mlp_net_valence(X_test_valence);
plotregression(t_test_valence, y_test_valence, ['Final test valence: 80 hidden Neurons']);
view(mlp_net_valence);

% Traces of other experiments
% max_neurons = 120;
% for i=5:5:max_neurons
%     mlp_net_valence = fitnet(i);
%     mlp_net_valence.divideParam.trainRatio = 0.8; 
%     mlp_net_valence.divideParam.valRatio = 0.2; 
%     mlp_net_valence.divideParam.testRatio = 0;
%     mlp_net_valence.trainParam.showWindow = 0;
%     mlp_net_valence.trainParam.showCommandLine = 1;
%     mlp_net_valence.trainParam.lr = 0.1; 
%     mlp_net_valence.trainParam.epochs = 100;
%     mlp_net_valence.trainParam.max_fail = 15;
%     [mlp_net_valence, tr_valence] = train(mlp_net_valence, X_train_valence, t_train_valence);
%     
% 
%     y_test_valence = mlp_net_valence(X_test_valence);
%     plotregression(t_test_valence, y_test_valence, ['Final test valence: ' string(i)]);
% end


%% Part with RBF training for arousal

%Parameters for training
spread_ar = 1.07;
goal_ar = 0;
K_ar = 1200;
Ki_ar = 100; %in order to speed up the training instead of the default 50

rbf_arousal = newrb(X_train_arousal,t_train_arousal,goal_ar,spread_ar,K_ar,Ki_ar);

% Test RBF
y_test_arousal = rbf_arousal(X_test_arousal);
plotregression(t_test_arousal, y_test_arousal, 'Final test arousal with RBF');

%% Part with RBF training for valence

%Parameters for training
spread_va = 0.7;
goal_va = 0;
K_va = 1200;
Ki_va = 100; %in order to speed up the training instead of the default 50

rbf_valence = newrb(X_train_valence,t_train_valence,goal_va,spread_va, K_va, Ki_va);

% Test RBF
y_test_valence = rbf_valence(X_test_valence);
plotregression(t_test_valence, y_test_valence, 'Final test valence with RBF');