%% Clean
clear
close all
clc
format compact

%% Load the dataset
dataset = load('data/dataset.mat');
dataset = table2array(dataset.dataset);

%% Remove infinite values
is_inf = isinf(dataset);
[rows_inf, ~] = find(is_inf == 1);
dataset(rows_inf,:) = [];


%% Removing outliers
% I can ignore the first 2 lines because they don't have significant values
dataset = dataset(:, 3:end);
[initial_rows, ~] = size(dataset);
% I use the default method 'median' in rmoutliers
clean_dataset = rmoutliers(dataset);
[final_rows, ~] = size(clean_dataset);
ret = initial_rows - final_rows;
fprintf("%i outliers removed\n", ret);

%% Data Balancing

EXTRACT_VALENCE = 1;
EXTRACT_AROUSAL = 0;
BALANCE = 1;

% getting arousal and valence levels
arousal_level  = clean_dataset(:,1);
valence_level = clean_dataset(:,2);

% samples for arousal and valence
sample_arousal = groupcounts(arousal_level);
sample_valence = groupcounts(valence_level);

% plot the graph
figure("Name", "Sample for arousal before balancing");
bar(sample_arousal);
title("Sample for arousal before balancing");

fprintf("Data are unbalanced\n");

[~, min_arousal] = min(sample_arousal);
[~, max_arousal] = max(sample_arousal);

% plot the graph
figure("Name", "Sample for valence before balancing");
bar(sample_valence);
title("Sample for valence before balancing");

[~, min_valence] = min(sample_valence);
[~, max_valence] = max(sample_valence);

augmentation_factors = [0 0];

debug = clean_dataset;
possible_values = [];

possible_values(1) = debug(10,1);
possible_values(2) = debug(1,1);
possible_values(3) = debug(8,1);
possible_values(4) = debug(15,1);
possible_values(5) = debug(7,1);
possible_values(6) = debug(21,1);
possible_values(7) = debug(27,1);

rep = 40;
row_to_check = final_rows;

if BALANCE == 1
    for k = 1:rep
        for i = 1:row_to_check
            if (clean_dataset(i,1)==possible_values(min_arousal) && clean_dataset(i,2)~=possible_values(max_valence)) || (clean_dataset(i,1)~=possible_values(max_arousal) && clean_dataset(i,2)==possible_values(min_valence))
                % Selection of i-th row
                selected_row = clean_dataset(i,:);
                % Augmentation of the i-th row
                row_to_add = selected_row;
                % Selection of the augmentation factor
                augmentation_factors(1) = 0.95+(0.04)*rand;
                augmentation_factors(2) = 1.01+(0.04)*rand;
                j = round(0.51+(1.98)*rand);
                % Augmentation
                row_to_add(3:end) = selected_row(3:end).*augmentation_factors(j); 
                % Addition of the new sample, obtained through augmentation, to
                % the dataset
                clean_dataset = [clean_dataset; row_to_add];
            end
            
            if((clean_dataset(i,1)==possible_values(max_arousal) && clean_dataset(i,2)~=possible_values(min_valence)) || (clean_dataset(i,2)==possible_values(max_valence) && clean_dataset(i,1)~=possible_values(min_arousal)))
                clean_dataset(i,:)=[];
                %fprintf(" I am removing the row %i\n",i);
            end
        end
        samples_arousal = groupcounts(clean_dataset(:,1));
        samples_valence = groupcounts(clean_dataset(:,2));
    
        [~, min_arousal] = min(samples_arousal);
        [~, max_arousal] = max(samples_arousal);
    
        [~, min_valence] = min(samples_valence);
        [~, max_valence] = max(samples_valence);
    end
    fprintf(" Balancing ended\n");
    
    samples_arousal = groupcounts(clean_dataset(:,1));
    samples_valence = groupcounts(clean_dataset(:,2));
    figure("Name", "Samples for arousal after balancing");
    bar(samples_arousal);
    title("Samples for arousal after balancing");
    fprintf("Arousal data balanced\n");
    figure("Name", "Samples for valence after balancing");
    bar(samples_valence);
    title("Samples for valence after balancing");
    fprintf("Valence data balanced\n");
end

%% Features selection
features = clean_dataset(:,3:end);
target_arousal = clean_dataset(:,1);
target_valence = clean_dataset(:,2);

cv = cvpartition(target_arousal, 'holdout', 0.3);
idxTraining = training(cv);
idxTesting = test(cv);

x_train = features(idxTraining, :);
y_train_arousal = target_arousal(idxTraining, :);
y_train_valence = target_valence(idxTraining, :);

x_test = features(idxTesting, :);
y_test_arousal = target_arousal(idxTesting, :);
y_test_valence = target_valence(idxTesting, :);

sequentialfs_rep = 30;


%% Features extraction for Arousal
if EXTRACT_AROUSAL == 1
    features_arousal = [zeros(1,54); 1:54]';
    counter_feat_sel_arousal = zeros(54,1)';
    for i = 1:sequentialfs_rep
        fprintf("Iteration %i\n", i);

        c = cvpartition(y_train_arousal, 'k', 10);
        option = statset('display','iter','useParallel',true);
        [features_selected_for_arousal, ~]  = sequentialfs(@myfun, x_train, y_train_arousal, 'cv', c, 'opt', option, 'nFeatures', 5);

        % Fetch useful indexes from result of latter sequentialfs
       p=1; 
       for j = 1:54
            if features_selected_for_arousal(j) == 1
                counter_feat_sel_arousal(j) = counter_feat_sel_arousal(j) + 1;
            end
        end
    end


    fprintf("\n");
    fprintf("*** AROUSAL: "); 
    fprintf("\n");

    disp(features_arousal);
    fprintf("Sorting features");
    features_arousal = sortrows(features_arousal, 1, 'descend');
    disp(features_arousal);

    % Getting the 10 best arousal features
    arousal_best = features_arousal(1:10, 2);

    best_arousal_training.x_train = normalize(x_train(:, arousal_best));
    best_arousal_training.y_train = y_train_arousal';
    % Save struct
    save("data/training_arousal.mat", "best_arousal_training");

    best_arousal_testing.x_test = normalize(x_test(:, arousal_best));
    best_arousal_testing.y_test = y_test_arousal';
    save("data/testing_arousal.mat", "best_arousal_testing");
    fprintf("Arousal features saved\n");

end
%% Features extraction for valence

if EXTRACT_VALENCE == 1

    features_valence = [zeros(1,54); 1:54]';
    
    for i = 1:sequentialfs_rep
        fprintf("Iteration %i\n", i);
        
        c = cvpartition(y_train_valence, 'k', 10);
        option = statset('display','iter','useParallel',true);
        inmodel = sequentialfs(@myfun, x_train, y_train_valence, 'cv', c, 'opt', option, 'nFeatures', 5);
        
        % Fetch useful indexes from result of latter sequentialfs
       p=1; 
       for val = inmodel
            if val == 1
                features_valence(p, 1) = features_valence(p, 1) + 1;
                fprintf("Added %d\n",p);
            end
            p = p+1;
        end
    
    end
        
    fprintf("\n");
    fprintf("valence:"); 
    disp(features_valence);
    fprintf("\n");
    
    disp(features_valence);
    fprintf("Sorting features");
    features_valence = sortrows(features_valence, 1, 'descend');
    disp(features_valence);
    
    % Getting the 10 best valence features
    valence_best = features_valence(1:10, 2);
    best_valance_training.x_train = normalize(x_train(:, valence_best));
    best_valance_training.y_train = y_train_valence';
    % Save struct
    save("data/training_valence.mat", "best_valance_training");
    
    best_valance_testing.x_test = normalize(x_test(:, valence_best));
    best_valance_testing.y_test = y_test_valence';
    save("data/testing_valence.mat", "best_valance_testing");
    fprintf("valence features saved\n");
end

%% Save best-3 features arousal dataset
arousal_best3 = features_arousal(1:3, 2);
best3.x_train = normalize(x_train(:, arousal_best3));
best3.y_train = y_train_arousal';
best3.x_test = normalize(x_test(:, arousal_best3));
best3.y_test = y_test_arousal';
best3.best_features=arousal_best3;
best3.y_values= possible_values;
% Save struct
save("data/best3.mat", "best3");
fprintf("Best-3 arousal features saved\n");

%% Function for sequentialfs
function err = myfun(x_train, t_train, x_test, t_test)
    net = fitnet(60);
    net.trainParam.showWindow=0;
    % net.trainParam.showCommandLine=1;
    xx = x_train';
    tt = t_train';
    net = train(net, xx, tt);
    y=net(x_test'); 
    err = perform(net,t_test',y);
end