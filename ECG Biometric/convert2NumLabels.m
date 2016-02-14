function[number_ClassLabels] = convert2NumLabels(classLabels)
    %Converts different class labels in string format to numerical format
    numberOfClasses = max(size(unique(classLabels)));
    numberOfSamples = max(size(classLabels));
    classes = unique(classLabels);
    number_ClassLabels = zeros(numberOfSamples,1);

    for i=1:numberOfClasses
        number_ClassLabels(strcmp(classes{i},classLabels)) = i;    
    end
end