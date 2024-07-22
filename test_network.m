Dataset = imageDatastore('Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[Training_Data, Validation_Data] = splitEachLabel(Dataset, 0.7,'randomized');

Resized_Training_Data = augmentedImageDatastore([256 256], Training_Data);
Resized_Validation_Data = augmentedImageDatastore([256 256], Validation_Data);

loaded_Network = load('Face_Recognizer12.mat');
net = loaded_Network.net;

[Label, Probability] = classify(net, Resized_Validation_Data);
accuracy = mean(Label == Validation_Data.Labels)

index = randperm(numel(Validation_Data.Files), 12);
figure
for i = 1:12
    subplot(3,4,i)
    I = readimage(Validation_Data, index(i));
    imshow(I)
    actual = Validation_Data.Labels(index(i))
    label = Label(index(i));
    title(string(label) + ", " + num2str(100*max(Probability(index(i), :)), 3) + "% реальный: " + string(actual));
end