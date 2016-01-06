function[split1, split2] = splitSamples(matrix,ratioToGet)
%ratio to get is the ratio of split1
[m, ~] = size(matrix);
howManySamples = round(m * ratioToGet);
sampleRows = randsample(m,howManySamples);
split1 = matrix(sampleRows,:);

complementSampleRows = 1:m;
complementSampleRows(sampleRows) = [];
split2 = matrix(complementSampleRows,:);

end