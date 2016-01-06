%-----(ii)Statistical Classification-----
%[trainingSet, validationSet, testSet, classTraining, classValidation, classTest] = getSets(howManyTraining,howManyValid,howManyTest, featureMatrix,class1);
clear; load('sets.mat');
%*****LDC, QDC*****
disp('*****LDC, QDC*****');
%Normalizing
trainingSet = zscore(trainingSet); 
validationSet = zscore(validationSet);
testSet = zscore(testSet);
%Construct datasets
trainingData = dataset(trainingSet,classTraining);
validationData = dataset(validationSet,classValidation);
testData = dataset(testSet,classTest);

%LDC Linear Bayes Normal Classifier 
%validation data
trueValidationLabels = getlab(validationData);
w = ldc(trainingData);
predictedValidationLabels = validationData * w * labeld;
confmat(trueValidationLabels,predictedValidationLabels);
disp(['LDC validationSet correct rate: ' num2str(mean(strcmp(cellstr(predictedValidationLabels),cellstr(trueValidationLabels))))]);
%test data
trueTestLabels = getlab(testData);
w = ldc(trainingData);
predictedTestLabels = testData * w * labeld;
confmat(trueTestLabels,predictedTestLabels);
disp(['LDC testSet correct rate: ' num2str(mean(strcmp(cellstr(predictedTestLabels),cellstr(trueTestLabels))))]);

%QDC Quadratic Bayes Normal Classifier
%validation data
trueValidationLabels = getlab(validationData);
w = qdc(trainingData);
predictedValidationLabels = validationData * w * labeld;
confmat(trueValidationLabels,predictedValidationLabels);
disp(['QDC validationSet correct rate: ' num2str(mean(strcmp(cellstr(predictedValidationLabels),cellstr(trueValidationLabels))))]);
%test data
trueTestLabels = getlab(testData);
w = qdc(trainingData);
predictedTestLabels = testData * w * labeld;
confmat(trueTestLabels,predictedTestLabels);
disp(['QDC testSet correct rate: ' num2str(mean(strcmp(cellstr(predictedTestLabels),cellstr(trueTestLabels))))]);

 %ldc and qdc
 trainingErrorPlot_ldc = clevalf(trainingData,ldc,[],[],1);
 figure();
 plote(trainingErrorPlot_ldc);
 
 trainingErrorPlot_qdc  = clevalf(trainingData,qdc,[],[],1);
 figure();
 plote(trainingErrorPlot_qdc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('%%%%%%%%%%%%%%%');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%*****Parzen*****
%validation data
trueValidationLabels = getlab(validationData);
w = parzenc(trainingData);
predictedValidationLabels = validationData * w * labeld;
confmat(trueValidationLabels,predictedValidationLabels);
disp(['Parzen validationSet correct rate: ' num2str(mean(strcmp(cellstr(predictedValidationLabels),cellstr(trueValidationLabels))))]);
%test data
trueTestLabels = getlab(testData);
w = parzenc(trainingData);
predictedTestLabels = testData * w * labeld;
confmat(trueTestLabels,predictedTestLabels);
disp(['Parzen testSet correct rate: ' num2str(mean(strcmp(cellstr(predictedTestLabels),cellstr(trueTestLabels))))]);
%*****kNN*****
%validation data
trueValidationLabels = getlab(validationData);
w = knnc(trainingData);
predictedValidationLabels = validationData * w * labeld;
confmat(trueValidationLabels,predictedValidationLabels);
disp(['kNN validationSet correct rate: ' num2str(mean(strcmp(cellstr(predictedValidationLabels),cellstr(trueValidationLabels))))]);
%test data
trueTestLabels = getlab(testData);
w = knnc(trainingData);
predictedTestLabels = testData * w * labeld;
confmat(trueTestLabels,predictedTestLabels);
disp(['kNN testSet correct rate: ' num2str(mean(strcmp(cellstr(predictedTestLabels),cellstr(trueTestLabels))))]);

 
%parzen and knn
trainingErrorPlot_parzen = clevalf(trainingData,parzenc,[],[],1);
figure();
plote(trainingErrorPlot_parzen);
 
trainingErrorPlot_knn  = clevalf(trainingData,knnc,[],[],1);
figure();
plote(trainingErrorPlot_knn);
