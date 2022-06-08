clear
close all
clc
format compact

%% Preparation of Data

% Dataset load
dataset = load('..\datasets\dataset.mat'); 
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
figure("Name", "Sample for valence before balancing");
bar(sample_valence);
title("Sample for valence before balancing");

fprintf("Data are unbalanced\n");

[~, min_arousal] = min(sample_arousal);
[~, max_arousal] = max(sample_arousal);
[~, min_valence] = min(sample_valence);
[~, max_valence] = max(sample_valence);

augmentation_factors = [0 0];

debug = dataset;
possible_values = [];

possible_values(1) = debug(10,1);
possible_values(2) = debug(1,1);
possible_values(3) = debug(8,1);
possible_values(4) = debug(15,1);
possible_values(5) = debug(7,1);
possible_values(6) = debug(21,1);
possible_values(7) = debug(27,1);

rep = 80;
row_to_check = final_rows;

for k = 1:rep
    for i = 1:row_to_check
        if (dataset(i,1)==possible_values(min_arousal) && dataset(i,2)~=possible_values(max_valence)) || (dataset(i,1)~=possible_values(max_arousal) && dataset(i,2)==possible_values(min_valence))
            % Selection of i-th row
            selected_row = dataset(i,:);
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
            dataset = [dataset; row_to_add];
        end
        
        if((dataset(i,1)==possible_values(max_arousal) && dataset(i,2)~=possible_values(min_valence)) || (dataset(i,2)==possible_values(max_valence) && dataset(i,1)~=possible_values(min_arousal)))
            dataset(i,:)=[];
            %fprintf(" I am removing the row %i\n",i);
        end
    end
    samples_arousal = groupcounts(dataset(:,1));
    samples_valence = groupcounts(dataset(:,2));

    [~, min_arousal] = min(samples_arousal);
    [~, max_arousal] = max(samples_arousal);

    [~, min_valence] = min(samples_valence);
    [~, max_valence] = max(samples_valence);
end
fprintf(" Balancing ended\n");

samples_arousal = groupcounts(dataset(:,1));
samples_valence = groupcounts(dataset(:,2));
figure("Name", "Samples for arousal after balancing");
bar(samples_arousal);
title("Samples for arousal after balancing");
fprintf("Arousal data balanced\n");
figure("Name", "Samples for valence after balancing");
bar(samples_valence);
title("Samples for valence after balancing");
fprintf("Valence data balanced\n");
