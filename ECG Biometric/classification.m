load train_test.mat

num_train_label = convert2NumLabels(train_label);
num_test_label = convert2NumLabels(test_label);

bestcv = 0;
c = [10,50,100,1000];
gamma = [0.1,0.2,0.5,1,10,50];
for i = 1:size(c,2)
  for j = 1:size(gamma,2)
    cmd = ['-v 5' ' -c ' num2str(c(i)) ' -g ', num2str(gamma(j)) ' -q'];
    cv = svmtrain(num_train_label, train, cmd);
    disp(['Result: gamma = ' num2str(gamma(j)) ' , c = ' num2str(c(i)) ' , Accuracy = ' num2str(cv)]);    
  end
end


model = svmtrain(num_train_label,train, '-t 2 -c 100 -g 0.05 -q');
disp('Training Data');
[predicted_train_label, accuracy_train, decision_values_train] = svmpredict(num_train_label, train, model);

disp('Testing Data');
[predicted_test_label, accuracy_test, decision_values_test] = svmpredict(num_test_label, test, model);

