function[randomRows] = getRandomSamples(matrix,ratioToGet)

[m n] = size(matrix);
howManySamples = round(m * ratioToGet);
sampleRows = randsample(m,howManySamples);
randomRows = matrix(sampleRows,:);

end