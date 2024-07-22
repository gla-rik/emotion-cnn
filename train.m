Dataset = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[Training_Data, Validation_Data] = splitEachLabel(Dataset, 0.7,'randomized');

Resized_Training_Data = augmentedImageDatastore([256 256], Training_Data);
Resized_Validation_Data = augmentedImageDatastore([256 256], Validation_Data);

layers = [
imageInputLayer([256 256 3])
convolution2dLayer(5,8,Padding="same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,Stride=2)
convolution2dLayer(5,16,Padding="same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,Stride=2)
convolution2dLayer(5,32,Padding="same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,Stride=2)
convolution2dLayer(5,64,Padding="same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,Stride=2)
convolution2dLayer(5,64,Padding="same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,Stride=2)
convolution2dLayer(5,32,Padding="same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,Stride=2)
convolution2dLayer(5,16,Padding="same")
batchNormalizationLayer
reluLayer
maxPooling2dLayer(2,Stride=2)
convolution2dLayer(5,8,Padding="same")
batchNormalizationLayer
reluLayer
fullyConnectedLayer(5)
softmaxLayer
classificationLayer];


Training_Options = trainingOptions('sgdm', ...
    'MiniBatchSize', 4,  ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 4e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Validation_Data, ...
    'ValidationFrequency', 3, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(Resized_Training_Data, layers, Training_Options);

save('Face_Recognizer12.mat', 'net');