Results:
By using SVM with RBF Kernel with parameters: gamma = 0.02 and C = 1000, the results on training and testing data is as follows:
*Training Data: Accuracy = 99.246% (5002/5040) (classification)
*Testing Data: Accuracy = 91.6402% (866/945) (classification)
*EER(Equal Error Rate) = 2.9333%

![alt tag](https://github.com/a-ozbek/Machine-Learning/blob/master/ECG%20Biometric/trainTestEER.tif)




Helper Functions: 
*"remove_noisy_segments.m"
*"normalize.m"
*"filter_ecg.m"
*"convert2NumLabels.m"


Scripts:

"getMatrices.m": Reads the dataset files ands constucts training, testing feature matrices and training, testing labels. Writes these into train_test.mat file

"classification.m": Performs classification (SVM) by using training and testing matrices created before and shows the result.

"eer.m": Calculates the Equal Error Rate(EER). and shows the result

*"benchmark.m": Combination of the above three scripts, "getMatrices.m", "classification.m", "eer.m".

To get the results, you can run either "benchmark.m" by itself or "getMatrices.m", "classification.m", "eer.m" in this order. 

