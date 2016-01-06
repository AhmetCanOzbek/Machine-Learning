function[trainingData,validation_data,test_data] = myScaleSets(trainingData,validation_data,test_data)

minimums = min(trainingData, [], 1);
ranges = max(trainingData, [], 1) - minimums;

trainingData = (trainingData - repmat(minimums, size(trainingData, 1), 1)) ./ repmat(ranges, size(trainingData, 1), 1);

test_data = (test_data - repmat(minimums, size(test_data, 1), 1)) ./ repmat(ranges, size(test_data, 1), 1);
validation_data = (validation_data - repmat(minimums, size(validation_data, 1), 1)) ./ repmat(ranges, size(validation_data, 1), 1);
