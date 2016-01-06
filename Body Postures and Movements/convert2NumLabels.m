function[numClassLabels] = convert2NumLabels(classLabels)
%Converts different class labels in string format to numerical format
howManyClass = max(size(unique(classLabels)));
howManyFeatures = max(size(classLabels));
classes = unique(classLabels);
numClassLabels = zeros(howManyFeatures,1);

for i=1:howManyClass
    numClassLabels(strcmp(classes{i},classLabels)) = i;    
end

end