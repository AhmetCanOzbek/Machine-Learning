function [errorTestRatePercentage] = minimimDistanceToMeansClassifier(featureTrainingMatrix, classTrainingLabels, featureTestMatrix, classTestLabels)


numberOfTrainingSamples = size(featureTrainingMatrix,1);
numberOfTestSamples = size(featureTestMatrix,1);
howManyFeatures = size(featureTrainingMatrix,2);

howManyClasses = max(size(unique(classTrainingLabels)));
%An array containing the names of tha classes, array length is number of
%classes
classes = unique(classTrainingLabels);
classMeans = zeros(howManyClasses,howManyFeatures);

    %%%Training 
    %Getting the class means from Training data
    for i=1:howManyClasses
        %indices = find(classTrainingLabels==classes(i));
        indices = find(strcmp(classTrainingLabels,classes(i))==1);
        classMeans(i,:) = sum(featureTrainingMatrix(indices,:))/size(featureTrainingMatrix(indices,:),1);          
    end
           
    %Error in Test Data
    numberOfErrorsInTestData = 0;
    for i=1:numberOfTestSamples
       samplePoint = featureTestMatrix(i,:);
       classIndex = findClosestClassMean(samplePoint, classMeans, howManyClasses);
       %if(classTrainingLabels(i) ~= classes(classIndex))
       if(~strcmp(classTestLabels(i),classes(classIndex)))
           numberOfErrorsInTestData =  numberOfErrorsInTestData + 1;      
       end         
    end        
    errorTestRatePercentage = (numberOfErrorsInTestData / numberOfTestSamples)*100;
      
end

function[whichClass] = findClosestClassMean(samplePoint, classMeans, howManyClasses)

    distanceToMeans = zeros(howManyClasses,1);
    
    for i=1:howManyClasses        
        distanceToMeans(i) = distance(samplePoint,classMeans(i,:));        
    end    
    
    [dummy whichClass] =  min(distanceToMeans);
end

function [d] = distance(x1,x2)
    
    d = sqrt(sum((x1 - x2).^2));
end

