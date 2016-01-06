function [binaryFeatureMatrix] = turnCategoricalFeatureToBinary(categoricalFeature)

numberOfFeatures = size(categoricalFeature,1);
howManyCategory = size(unique(categoricalFeature),1);
categories = unique(categoricalFeature);

binaryFeatureMatrix = zeros(numberOfFeatures,howManyCategory);

    for i=1:howManyCategory    
         binaryFeatureMatrix(:,i) = strcmp(categoricalFeature,categories(i));       
    end
end