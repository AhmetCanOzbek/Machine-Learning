%Construct the featureMatrix
load('dataSet5');
featureMatrix = [age body_mass_index turnCategoricalFeatureToBinary(gender)...
    how_tall_in_meters weight x1 x2 x3 x4 y1 y2 y3 y4 z1 z2 z3 z4];

 featureMatrix = deleteRows(featureMatrix,122077);
 class1 = deleteRows(class1, 122077);
 minimimDistanceToMeansClassifier(featureMatrix,class1);

howManyTraining = 3000;
howManyValid = 3000;
howManyTest = 4000;
[trainingSet, validationSet, testSet, classTraining, classValidation, classTest, temp] = getSets(howManyTraining,howManyValid,howManyTest, featureMatrix,class1);
