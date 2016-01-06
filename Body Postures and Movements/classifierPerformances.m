%-----Default Systems-----
%[trainingSet, validationSet, testSet, classTraining, classValidation, classTest] = getSets(howManyTraining,howManyValid,howManyTest, featureMatrix,class1);
clear; load('sets.mat');
disp('***Default Systems***');
%*****Random assignment to classes*****
load('dataSet5.mat');
classes = unique(class1);
howManyClass = max(size(unique(class1)));
howManySamples = max(size(class1,1));
for i=1:howManyClass
    r(i) = sum(strcmp(class1,classes(i)));
end
%randomly assign class labels with priors
randomAssignedLabels = randsample(howManyClass,howManySamples,true,r);
correctRate = mean(convert2NumLabels(class1) == randomAssignedLabels);
disp(['***Random Assignment To Classes Correct Rate: ' num2str(correctRate)]);

%*****Baseline System: Minimum distance to means classifier*****
% trainingSet = myPhiMachine(trainingSet);
% validationSet = myPhiMachine(validationSet);
% testSet = myPhiMachine(testSet);
disp('*****Baseline System: Minimum distance to means classifier*****');
%Without normalizing
disp('*Without Normalizing');
disp(['Correct Rate in Training Set: ' num2str(100-minimimDistanceToMeansClassifier(trainingSet,classTraining,trainingSet,classTraining)) '%']);
disp(['Correct Rate in Validation Set: ' num2str(100-minimimDistanceToMeansClassifier(trainingSet,classTraining,validationSet,classValidation)) '%']);
disp(['Correct Rate in Test Set: ' num2str(100-minimimDistanceToMeansClassifier(trainingSet,classTraining,testSet,classTest)) '%']);
%With normalizing
disp('*With Normalizing');
disp(['Correct Rate in Training Set: ' num2str(100-minimimDistanceToMeansClassifier(zscore(trainingSet),classTraining,zscore(trainingSet),classTraining)) '%']);
disp(['Corrext Rate in Validation Set: ' num2str(100-minimimDistanceToMeansClassifier(zscore(trainingSet),classTraining,zscore(validationSet),classValidation)) '%']);
disp(['Correct Rate in Test Set: ' num2str(100-minimimDistanceToMeansClassifier(zscore(trainingSet),classTraining,zscore(testSet),classTest)) '%']);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('');
%%%%%%%%%%%%%%%%%%%%%%%%%%%
 

 