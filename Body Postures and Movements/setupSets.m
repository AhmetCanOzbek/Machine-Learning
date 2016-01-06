%Construct the featureMatrix
load('dataSet5');
featureMatrix = [age body_mass_index turnCategoricalFeatureToBinary(gender)...
    how_tall_in_meters weight x1 x2 x3 x4 y1 y2 y3 y4 z1 z2 z3 z4];
 %delete the samples with missing feature 122077
 [rowNumber columnNumber] = find(isnan(featureMatrix) == 1); 
 featureMatrix = deleteRows(featureMatrix,rowNumber);
 class1 = deleteRows(class1, rowNumber); 
 %Arranging training, validation, and test sets
 howManyTraining = 6000;
 howManyValid = 2000;
 howManyTest = 2000;
 [trainingSet, validationSet, testSet, classTraining, classValidation, classTest] = getSets(howManyTraining,howManyValid,howManyTest, featureMatrix,class1);
 
 save('sets','trainingSet', 'validationSet', 'testSet', 'classTraining', 'classValidation', 'classTest');