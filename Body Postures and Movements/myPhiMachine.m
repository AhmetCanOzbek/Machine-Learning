function[newMatrix] = myPhiMachine(featureMatrix)

[howManySamples, howManyFeatures] = size(featureMatrix);
counter = 0;
for i=1:howManyFeatures
    for j=i:howManyFeatures
        counter = counter + 1;
        newMatrix(:,counter) = featureMatrix(:,i) .* featureMatrix(:,j);        
    end    
end

newMatrix = [ones(howManySamples,1) featureMatrix(:,1:howManyFeatures)  newMatrix];
end