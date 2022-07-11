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

MLP_AROUSAL = 1;
MLP_VALENCE = 1;
RBFN_AROUSAL = 1;
RBFN_VALENCE = 1;
TESTING = 0;

%% MLP for Arousal

% Experiments for determining the hidden layer size
if TESTING == 1 
    max_neurons_1 = 120;
    R_saved = 0;
    hiddenLayerSize_arousal = 0;
    
    for i=5:5:max_neurons_1    
         mlp_net_arousal = fitnet(i);
         mlp_net_arousal.divideParam.trainRatio = 0.7;
         mlp_net_arousal.divideParam.testRatio = 0.1; 
         mlp_net_arousal.divideParam.valRatio = 0.2;
         mlp_net_arousal.trainParam.showWindow = 0;
         mlp_net_arousal.trainParam.showCommandLine = 1;
         mlp_net_arousal.trainParam.lr = 0.1; 
         mlp_net_arousal.trainParam.epochs = 150;
         mlp_net_arousal.trainParam.max_fail = 10;
         [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, x_train_arousal, y_train_arousal);

         test_output_arousal = mlp_net_arousal(x_test_arousal);
         v = figure;
         plotregression(y_test_arousal, test_output_arousal);
         
         str = v.Children(3).Title.String;
         ind=find(str=='=');
         R_in_str =str(ind+1:end);
         R=str2double(R_in_str);
         if R_saved < R
             R_saved = R;
             hiddenLayerSize_arousal = i;
         end
    end
    
    fprintf("Max R value saved: %d for hiddenLayerSize %d \n", R, hiddenLayerSize_arousal);
    
    max_neurons = 120;
    R_saved = 0;
    hiddenLayerSize_valence = 0;
    
    for i=5:5:max_neurons
        mlp_net_valence = fitnet(i);
        mlp_net_valence.divideParam.trainRatio = 0.7; 
        mlp_net_valence.divideParam.valRatio = 0.2; 
        mlp_net_valence.divideParam.testRatio = 0.1;
        mlp_net_valence.trainParam.showWindow = 0;
        mlp_net_valence.trainParam.showCommandLine = 1;
        mlp_net_valence.trainParam.lr = 0.1; 
        mlp_net_valence.trainParam.epochs = 100;
        mlp_net_valence.trainParam.max_fail = 15;
        [mlp_net_valence, tr_valence] = train(mlp_net_valence, x_train_valence, y_train_valence);
        
        test_output_valence = mlp_net_valence(x_test_valence);
        v = figure;
        plotregression(y_test_valence, test_output_valence);
        
        str = v.Children(3).Title.String;
        ind=find(str=='=');
        R_in_str =str(ind+1:end);
        R=str2double(R_in_str);
        if R_saved < R
             R_saved = R;
             hiddenLayerSize_valence = i;
         end
    end
    
    fprintf("Max R value saved: %d for hidden layer %d \n", R, hiddenLayerSize_valence);

end100

if MLP_AROUSAL == 1
    % Creation of MLP for arousal
    hiddenLayerSize_arousal = 25;
    mlp_arousal = fitnet(hiddenLayerSize_arousal);
    mlp_arousal.divideParam.trainRatio = 0.7;
    mlp_arousal.divideParam.testRatio = 0.1;
    mlp_arousal.divideParam.valRatio = 0.2;
    mlp_arousal.divideParam.lr = 0.1;
    mlp_arousal.trainParam.epochs = 110;
    mlp_arousal.trainParam.max_fail = 10;
    
    mlp_arousal.trainParam.showCommandLine = 1;
    %mlp_arousal.trainParam.showWindow=0;

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
    % Creation of MLP for valence
    hiddenLayerSize_valence = 45;
    mlp_valence = fitnet(hiddenLayerSize_valence);
    mlp_valence.divideParam.trainRatio = 0.7;
    mlp_valence.divideParam.testRatio = 0.1;
    mlp_valence.divideParam.valRatio = 0.2;
    mlp_valence.trainParam.lr = 0.1; 
    mlp_valence.trainParam.epochs = 110;
    mlp_net_valence.trainParam.max_fail = 15;
    
    %mlp_valence.trainParam.showWindow=0;
    mlp_valence.trainParam.showCommandLine = 1;

    % Training
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
    spread_ar = 1.07;
    K_ar = 500;
    Ki_ar = 50;

    rbf_arousal = newrb(x_train_arousal,y_train_arousal,goal_ar,spread_ar,K_ar,Ki_ar);
    view (rbf_arousal);
    %Test
    test_output_arousal_rbf = rbf_arousal(x_test_arousal);
    figure(5)
    %Plot regression
    plotregression(y_test_arousal, test_output_arousal_rbf, 'Final test arousal with RBF');
end

%% RBFN for Valence

if RBFN_VALENCE == 1
    %Creation of RBFN
    goal_va = 0;
    spread_va = 0.7;
    K_va = 500;
    Ki_va = 50;
    
    rbf_valence = newrb(x_train_valence,y_train_valence,goal_va,spread_va, K_va, Ki_va);
    view (rbf_valence);
    %Test
    test_output_valence_rbf = rbf_valence(x_test_valence);
    %Plot regression
    figure(6)
    plotregression(y_test_valence, test_output_valence_rbf, 'Final test valence with RBF');
end