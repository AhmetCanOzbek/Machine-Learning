%[trainingSet, validationSet, testSet, classTraining, classValidation, classTest] = getSets(howManyTraining,howManyValid,howManyTest, featureMatrix,class1);
%(iii)-----SVM-----
clear;
load('sets.mat');
% %*****SVM with default paremeters*****
% model = svmtrain(convert2NumLabels(classTraining),trainingSet,'-q');
% [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest), testSet, model);
% disp(['1*****SVM with default parameters without zscore: ' num2str(mean(predicted_label == convert2NumLabels(classTest)))]);
% 
% %*****SVM with default paremeters with zscore()*****
% model = svmtrain(convert2NumLabels(classTraining),zscore(trainingSet),'-q');
% [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest), zscore(testSet), model);
% disp(['2*****SVM with default parameters with zscore(): ' num2str(mean(predicted_label == convert2NumLabels(classTest)))]);
% 
% %*****SVM with parameters*****
% model = svmtrain(convert2NumLabels(classTraining),zscore(trainingSet),'-t 1 -g 200 -q');
% [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest), zscore(testSet), model,'');
% disp(mean(predicted_label == convert2NumLabels(classTest)));
% disp(['3*****SVM with parameters: ' num2str(mean(predicted_label == convert2NumLabels(classTest)))]);
% 
% %*****SVM with scaling*****
% [scaledTraining, scaledValid, scaledTest] = myScaleSets(trainingSet,validationSet,testSet);
% model = svmtrain(convert2NumLabels(classTraining),scaledTraining,'-t 2 -g 2000 -q');
% [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest), scaledTest, model,'');
% disp(mean(predicted_label == convert2NumLabels(classTest)));
% disp(['4*****SVM with scaling: ' num2str(mean(predicted_label == convert2NumLabels(classTest)))]);

% %*****SVM with parameters*****
% [scaledTraining, scaledValid, scaledTest] = myScaleSets(trainingSet,validationSet,testSet);
% model = svmtrain(convert2NumLabels(classTraining),scaledTraining,'-s 0 -c 500 -t 2 -g 100 -q');
% [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest),scaledTest, model,'');
% disp(mean(predicted_label == convert2NumLabels(classTest)));
% disp(['3*****SVM with parameters: ' num2str(mean(predicted_label == convert2NumLabels(classTest)))]);

%WITH LOOPS
[scaledTraining, scaledValid, scaledTest] = myScaleSets(trainingSet,validationSet,testSet);
%for linear(-t 0)
c = [50,500,5000,10000];
resultLinearMatrix = zeros(size(c,2),1);
for i=1:size(c,2)    
        model = svmtrain(convert2NumLabels(classTraining),scaledTraining,['-t 1' ' -c ' num2str(c(i)) ' -q' ]);
        [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest), scaledValid, model);
        resultLinearMatrix(i) = mean(predicted_label == convert2NumLabels(classTest));       
end

%for Polynomial(-t 1)
c = [50,500,5000];
gamma = [10,50,100,500,1000,2500,5000];
degree = [2,3,4];
resultMatrixPoly = zeros(size(c,2),size(gamma,2));
for i=1:size(degree,2)
    for j=1:size(gamma,2)
        model = svmtrain(convert2NumLabels(classTraining),scaledTraining,['-t 1 ' ' -c 100 ' ' -g ' num2str(gamma(j)) ' -d ' num2str(degree(i)) ' -q' ]);
        [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest), scaledValid, model);
        resultMatrixPoly(i,j) = mean(predicted_label == convert2NumLabels(classTest));
    end    
end

% %for Gaussian (RBF) (-t 2)
% c = [50,500,5000];
% gamma = [10,50,100,500,1000,2500,5000];
% resultMatrixRBF = zeros(size(c,2),size(gamma,2));
% for i=1:size(c,2)
%     for j=1:size(gamma,2)
%         model = svmtrain(convert2NumLabels(classTraining),scaledTraining,['-t 2' ' -c ' num2str(c(i)) ' -g ' num2str(gamma(j)) ' -q' ]);
%         [predicted_label, accuracy, decision_values] = svmpredict(convert2NumLabels(classTest), scaledValid, model);
%         resultMatrixRBF(i,j) = mean(predicted_label == convert2NumLabels(classTest));
%     end    
% end

