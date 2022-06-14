%% Preparation of data
clear
close all
clc


% Config variables
% who_to_train: 0-> only two_classes CNN, 1->only four_classes CNN, 2->both
who_to_train = 2; 


% Preparing labels
anger_cell = cell(1,250);
anger_cell(:) = {'Anger'};
disgust_cell = cell(1,250);
disgust_cell(:) = {'Disgust'};
fear_cell = cell(1,250);
fear_cell(:) = {'Fear'};
happiness_cell = cell(1,250);
happiness_cell(:) = {'Happiness'};
emotions_labels_four = categorical([anger_cell disgust_cell fear_cell happiness_cell]);
emotions_labels_two = categorical([disgust_cell fear_cell]);

% Base image to get the proper size for the first layer of the CNN
base_img = imread('datasets/images/10011.jpg');
base_img_size = size(base_img);

% Creation of the two datastore, first one is for a 4-classes
% classification, second for a 2-classes classification
imds_four = imageDatastore('datasets/images/images_to_use_experiment_2', 'labels', emotions_labels_four);
imds_two = imageDatastore('datasets/images/images_to_use_experiment_2_two_classes', 'labels', emotions_labels_two);

% Data split in training and testing set
fracTrain = 0.8;
[imdsTrain_two,imdsTest_two] = splitEachLabel(imds_two,fracTrain,'randomize');
[imdsTrain_four,imdsTest_four] = splitEachLabel(imds_four,fracTrain,'randomize');

%Method for image augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-20,20],...
    'RandXReflection', true, ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);

augimds_two = augmentedImageDatastore(base_img_size,imdsTrain_two,'DataAugmentation',imageAugmenter);
augimds_four = augmentedImageDatastore(base_img_size,imdsTrain_four,'DataAugmentation',imageAugmenter);


%CNN layers for two-classes classification (fear and disgust emotions)
leaky_layers_two = [
    imageInputLayer(size(base_img))
    
    convolution2dLayer(16,6,'Stride',4,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(8,12,'Stride',2,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(40)
    fullyConnectedLayer(2)
    
    softmaxLayer
    
    classificationLayer
];

%CNN layers for four-classes classification 
leaky_layers_four = [
    imageInputLayer(size(base_img))
    
    convolution2dLayer(16,8,'Stride',4,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(8,16,'Stride',2,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2,'Stride', 1)
    
    convolution2dLayer(4,32,'Stride',2,'Padding','same')
    batchNormalizationLayer
    leakyReluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(100)
    fullyConnectedLayer(100)
    fullyConnectedLayer(4)
    
    softmaxLayer
    
    classificationLayer
];


% Traces of other experiments
% ---------------------------
% layers = [
%     imageInputLayer(size(base_img))
%     
%     convolution2dLayer(10,24,'Stride',4,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(5,48,'Stride',2,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,96,'Stride',1,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,96,'Stride',1,'Padding','same')
%     batchNormalizationLayer
%     leakyReluLayer
%     
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,96,'Stride',1,'Padding','same')
%     batchNormalizationLayer
%     leakyReluLayer
%     
%     fullyConnectedLayer(100)
%     fullyConnectedLayer(4)
%     
%     softmaxLayer
%     
%     classificationLayer
% ];


%% Training

%Training options for two CNNs
options_sgdm_two = trainingOptions('sgdm', ...
'InitialLearnRate', 0.01, ...
'MiniBatchSize', 40, ...
'MaxEpochs', 200, ...
'Shuffle','every-epoch', ...
'ValidationData', imdsTest_two, ...
'ValidationFrequency', 20, ...
'Verbose', false, ...
'Plots', 'training-progress');

options_sgdm_four = trainingOptions('sgdm', ...
'InitialLearnRate', 0.01, ...
'MiniBatchSize', 40, ...
'MaxEpochs', 100, ...
'Shuffle','every-epoch', ...
'ValidationData', imdsTest_four, ...
'ValidationFrequency', 20, ...
'Verbose', false, ...
'Plots', 'training-progress');

if who_to_train == 2
    net_sgdm_leaky_two = trainNetwork(augimds_two,leaky_layers_two,options_sgdm_two);
    net_sgdm_leaky_four = trainNetwork(augimds_four,leaky_layers_four,options_sgdm_four);
elseif who_to_train == 1
    net_sgdm_leaky_four = trainNetwork(augimds_four,leaky_layers_four,options_sgdm_four);
else
    net_sgdm_leaky_two = trainNetwork(augimds_two,leaky_layers_two,options_sgdm_two);
end


% Traces of other experiments
% ---------------------------
% options_adam = trainingOptions('adam', ...
% 'InitialLearnRate', 0.01, ...
% 'MiniBatchSize', 20, ...
% 'MaxEpochs', 40, ...
% 'Shuffle','every-epoch', ...
% 'ValidationData', imdsTest, ...
% 'ValidationFrequency', 4, ...
% 'Verbose', false, ...
% 'Plots', 'training-progress');
%
% options_rmsprop = trainingOptions('rmsprop', ...
% 'InitialLearnRate', 0.01, ...
% 'MiniBatchSize', 20, ...
% 'MaxEpochs', 40, ...
% 'Shuffle','every-epoch', ...
% 'ValidationData', imdsTest, ...
% 'ValidationFrequency', 20, ...
% 'Verbose', false, ...
% 'Plots', 'training-progress');
% net_sgdm = trainNetwork(augimds,layers,options_sgdm);
% net_adam = trainNetwork(augimds,layers,options_adam);
% net_rmsprop = trainNetwork(augimds,layers,options_rmsprop);
% net_adam_leaky = trainNetwork(augimds,leaky_layers,options_adam);
% net_rmsprop_leaky = trainNetwork(augimds,leaky_layers,options_rmsprop);
%% Performance Assessment

%Plotting of confusion matrix for given test set
if who_to_train == 2
    predLabels_sgdm_leaky_two = classify(net_sgdm_leaky_two,imdsTest_two);
    predLabels_sgdm_leaky_four = classify(net_sgdm_leaky_four,imdsTest_four);
    testLabels_two = imdsTest_two.Labels;
    testLabels_four = imdsTest_four.Labels;
    
    accuracy_sgdm_leaky_two = sum(predLabels_sgdm_leaky_two == testLabels_two)/numel(testLabels_two);
    fprintf('Accuracy sgdm leaky 2 classes is %8.2f%%\n',accuracy_sgdm_leaky_two*100);
    accuracy_sgdm_leaky_four = sum(predLabels_sgdm_leaky_four == testLabels_four)/numel(testLabels_four);
    fprintf('Accuracy sgdm leaky 4 classes is %8.2f%%\n',accuracy_sgdm_leaky_four*100);
    
    
    figure
    plotconfusion(testLabels_two, predLabels_sgdm_leaky_two, 'Confusion Matrix 2 Classes');

    figure
    plotconfusion(testLabels_four, predLabels_sgdm_leaky_four, 'Confusion Matrix 4 Classes');

elseif who_to_train == 1
    predLabels_sgdm_leaky_four = classify(net_sgdm_leaky_four,imdsTest_four);
    testLabels_four = imdsTest_four.Labels;
    accuracy_sgdm_leaky_four = sum(predLabels_sgdm_leaky_four == testLabels_four)/numel(testLabels_four);
    fprintf('Accuracy sgdm leaky 4 classes is %8.2f%%\n',accuracy_sgdm_leaky_four*100);
    figure
    plotconfusion(testLabels_four, predLabels_sgdm_leaky_four, 'Confusion Matrix 4 Classes');
else
    predLabels_sgdm_leaky_two = classify(net_sgdm_leaky_two,imdsTest_two);
    testLabels_two = imdsTest_two.Labels;
    accuracy_sgdm_leaky_two = sum(predLabels_sgdm_leaky_two == testLabels_two)/numel(testLabels_two);
    fprintf('Accuracy sgdm leaky 2 classes is %8.2f%%\n',accuracy_sgdm_leaky_two*100);
    figure
    plotconfusion(testLabels_two, predLabels_sgdm_leaky_two, 'Confusion Matrix 2 Classes');
end


% Traces of other experiments
% ---------------------------
% predLabels_sgdm = classify(net_sgdm,imdsTest);
% predLabels_adam = classify(net_adam,imdsTest);
% predLabels_rmsprop = classify(net_rmsprop,imdsTest);
% predLabels_adam_leaky = classify(net_adam_leaky,imdsTest);
% predLabels_rmsprop_leaky = classify(net_rmsprop_leaky,imdsTest);
% accuracy_sgdm = sum(predLabels_sgdm == testLabels)/numel(testLabels);
% fprintf('Accuracy sgdm is %8.2f%%\n',accuracy_sgdm*100);
% accuracy_adam = sum(predLabels_adam == testLabels)/numel(testLabels);
% fprintf('Accuracy adam is %8.2f%%\n',accuracy_adam*100);
% accuracy_rmsprop = sum(predLabels_rmsprop == testLabels)/numel(testLabels);
% fprintf('Accuracy rmsprop is %8.2f%%\n',accuracy_rmsprop*100);
% accuracy_adam_leaky = sum(predLabels_adam_leaky == testLabels)/numel(testLabels);
% fprintf('Accuracy adam leaky is %8.2f%%\n',accuracy_adam_leaky*100);
% accuracy_rmsprop_leaky = sum(predLabels_rmsprop_leaky == testLabels)/numel(testLabels);
% fprintf('Accuracy rmsprop leaky is %8.2f%%\n',accuracy_rmsprop_leaky*100);