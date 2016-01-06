%-----(i)Distribution Free Classification-----
%[trainingSet, validationSet, testSet, classTraining, classValidation, classTest] = getSets(howManyTraining,howManyValid,howManyTest, featureMatrix,class1);
 clear; load('sets.mat');
trainingSet = myPhiMachine(trainingSet);
validationSet = myPhiMachine(validationSet);
testSet = myPhiMachine(testSet);
 %*****Perceptron*****
 disp('*****Perceptron*****');
 [train_targetsP, ~] = multiclass(zscore(trainingSet)', double(convert2NumLabels(classTraining))', zscore(trainingSet)', '[''OAA'', 0, ''Perceptron'', [10000]]');
 validation_error_Percept_ova = mean(convert2NumLabels(classTraining)'==train_targetsP);
 disp(['Perceptron Correct Rate on Training Set:' num2str(validation_error_Percept_ova)]);
 
 [validation_targetsP, ~] = multiclass(zscore(trainingSet'), double(convert2NumLabels(classTraining))', zscore(validationSet'), '[''OAA'', 0, ''Perceptron'', [10000]]');
 validation_error_Percept_ova = mean(convert2NumLabels(classValidation)'==validation_targetsP);
 disp(['Perceptron Correct Rate on Validation Set:' num2str(validation_error_Percept_ova)]);
 
 [test_targets, ~] = multiclass(zscore(trainingSet'), convert2NumLabels(classTraining)', zscore(testSet'), '[''OAA'', 0, ''Perceptron'', [10000]]');
 test_error_Percept_ova = mean(convert2NumLabels(classTest)'==test_targets);
 disp(['Perceptron Correct Rate on Test Set:' num2str(test_error_Percept_ova)]); 
 
 %Normalizing
 trainingSet = zscore(trainingSet); 
 validationSet = zscore(validationSet);
 testSet = zscore(testSet);
 %Construct datasets
 trainingData = dataset(trainingSet,classTraining);
 validationData = dataset(validationSet,classValidation);
 testData = dataset(testSet,classTest);
 %Analyze Dimensionality
 ErrorTrainPerceptron = clevalf(trainingData,perlc,[],[],1);
 figure();
 plote(ErrorTrainPerceptron); 
 
 
 %*****MSE*****
 disp('*****MSE*****');
 [training_targets, ~] = multiclass(zscore(trainingSet)', double(convert2NumLabels(classTraining))', zscore(trainingSet)', '[''OAA'', 0, ''LS'', []]');
 validation_error_MSE_ova = mean(convert2NumLabels(classTraining)'==training_targets);
 disp(['MSE Correct Rate on Training Set:' num2str(validation_error_MSE_ova)]);
 
  [validation_targets, ~] = multiclass(zscore(trainingSet)', double(convert2NumLabels(classTraining))', zscore(validationSet)', '[''OAA'', 0, ''LS'', []]');
 validation_error_MSE_ova = mean(convert2NumLabels(classValidation)'==validation_targets);
 disp(['MSE Correct Rate on Validation Set:' num2str(validation_error_MSE_ova)]);
 
 [test_targets, ~] = multiclass(zscore(trainingSet)', double(convert2NumLabels(classTraining))', zscore(testSet)', '[''OAA'', 0, ''LS'', []]');
 test_error_MSE_ova = mean(convert2NumLabels(classTest)'==test_targets);
 disp(['MSE Correct Rate on Test Set:' num2str(test_error_MSE_ova)]); 