function[trainingSet, validationSet, testSet, classTraining, classValidation, classTest] = getSets(howManyTraining,howManyValid,howManyTest, featureMatrix,classLabels)

total = howManyTraining + howManyValid + howManyTest;
howManyClass = size(unique(classLabels),1);
howManyFeatures = size(featureMatrix,1);
classes = unique(classLabels);
%getting ratios
r = zeros(howManyClass,1);
trainIndices = []; 
validIndices = [];
testIndices = [];
for i=1:howManyClass
    r(i) = sum(strcmp(classLabels,classes{i})) / howManyFeatures;   
    temp = randsample(find(strcmp(classLabels,classes{i})==1),round(total * r(i)));   
    [y] = mySplit(temp,[round(howManyTraining*r(i)) round(howManyValid*r(i)) round(total * r(i))-(round(howManyTraining*r(i))+round(howManyValid*r(i)))]);
    newTrainIndices = y{1};
    newValidIndices = y{2};
    newTestIndices = y{3};    
    trainIndices = [trainIndices newTrainIndices'];
    validIndices = [validIndices newValidIndices'];
    testIndices = [testIndices newTestIndices']; 
    
end
%training
trainingSet = featureMatrix(trainIndices,:); classTraining = classLabels(trainIndices,:);
validationSet = featureMatrix(validIndices,:); classValidation = classLabels(validIndices,:);
testSet = featureMatrix(testIndices,:); classTest = classLabels(testIndices,:);

end

function[y] = mySplit(input,lengths)

start=0; endp=0;
howManyPieces = max(size(lengths));
y = [];

for i=1:howManyPieces
    start = endp + 1;
    endp = start + lengths(i) -1;    
    y{i} = input(start:endp);        
end 

end