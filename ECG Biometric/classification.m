load train_test.mat

num_train_label = convert2NumLabels(train_label);
num_test_label = convert2NumLabels(test_label);


%Normalize train and test data


%model = svmtrain(num_train_label,train, '-t 2 -c 50 -g 0.2 -v 5');

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

% bestcv = 0;
% for log2c = 0:3,
%   for log2g = -4:1,
%     cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), ' -q'];
%     cv = svmtrain(num_train_label, train, cmd);
%     if (cv >= bestcv),
%       bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
%     end
%     fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
%   end
% end

% %This is the one I run
% bestcv = 0;
% c = [1000,2000,500,100];
% gamma = [0.005,0.02,0.1,0.2,0.5,1];
% for i = 1:size(c,2)
%   for j = 1:size(gamma,2)
%     cmd = ['-t 2 -v 5' ' -c ' num2str(c(i)) ' -g ', num2str(gamma(j)) ' -q'];
%     cv = svmtrain(num_train_label, train, cmd);
%     disp(['Result: gamma = ' num2str(gamma(j)) ' , c = ' num2str(c(i)) ' , Accuracy = ' num2str(cv)]);    
%   end
% end

C = 1000;
gamma = 0.02;
disp(['SVM Parameters: ' 'C = ' num2str(C) ' , gamma = ' num2str(gamma)]);
model = svmtrain(num_train_label,train,['-t 2 -c ' num2str(C) ' -g ' num2str(gamma) ' -b 1 -q']);
disp('Training Data');
[predicted_train_label, accuracy_train, prob_estimates_train] = svmpredict(num_train_label, train, model, '-b 1');

disp('Testing Data');
[predicted_test_label, accuracy_test, prob_estimates_test] = svmpredict(num_test_label, test, model, '-b 1');

