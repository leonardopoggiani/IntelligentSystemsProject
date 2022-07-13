%% Clean
clear
close all
clc
format compact

%% Load data
% if we want to classify just two classes instead of all four classes (more
% computationally demanding) just set this variable
CLASSIFICATION = 5;

if CLASSIFICATION == 0
    image_data = imageDatastore("data/images/selected/classification_2_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 1
        image_data = imageDatastore("data/images/selected/classification_4_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 2
        image_data = imageDatastore("data/images/noSelected/classification_2_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 3
        image_data = imageDatastore("data/images/noSelected/classification_4_classes/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 4
        image_data = imageDatastore("data/images/1000_images/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
elseif CLASSIFICATION == 5
        image_data = imageDatastore("data/images/500_images/",'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end

% 70 per training, 20 per validation, 10 per test
[data_train, data_valtest] = splitEachLabel(image_data, 0.7, 'randomized');
[data_validation, data_test] = splitEachLabel(image_data, 0.2, 'randomized');
numClasses = numel(categories(data_train.Labels));

%% Alexnet

net = alexnet;
input_size = net.Layers(1).InputSize;

% Extract all the layers except the last 3
original_layers = net.Layers(1:end-3);

layers = [
    original_layers
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augmented_image_data_train = augmentedImageDatastore(input_size(1:2), data_train, 'DataAugmentation', imageAugmenter);
augmented_image_data_validation = augmentedImageDatastore(input_size(1:2), data_validation);
augmented_image_data_test = augmentedImageDatastore(input_size(1:2), data_test);

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData',augmented_image_data_validation, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% Training

new_CNN = trainNetwork(augmented_image_data_train, layers, options);
hold on

%% Training

hold on
[res_train, scores_train] = classify(new_CNN, augmented_image_data_train);
target_train = data_train.Labels;
accuracy_train = mean (target_train==res_train);
plotconfusion(target_train, res_train)

%% Validation

[res_val, scores_val] = classify(new_CNN, augmented_image_data_validation);
target_val = data_validation.Labels;
accuracy_val= mean (target_val==res_val);
plotconfusion(target_val, res_val)

%% Testing
[res_test, scores_test] = classify(new_CNN, augmented_image_data_test);
target_test = data_test.Labels;
accuracy_test = mean(target_test==res_test);
plotconfusion(target_test, res_test)