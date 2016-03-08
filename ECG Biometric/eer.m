%Script for calculating EER (Equal Error Rate)

%Parameters
numberOfSubjects = 63;
numberOfHeartbeats = 15;

%Identification
majority_predicted_test_label = zeros(63,1);
majority_test_label = zeros(63,1);
for i=1:numberOfSubjects
    majority_predicted_test_label(i) =  mode(predicted_test_label(15*(i-1)+1 : 15*(i-1)+15));  
    majority_test_label(i) =  mode(num_test_label(15*(i-1)+1 : 15*(i-1)+15));
end
identification_error = (sum(majority_predicted_test_label ~= majority_test_label)) / numberOfSubjects;

disp(['Identification Error After Majority Vote: ' num2str(identification_error)]);


targets = [];
for i=1:numberOfSubjects
    row = zeros(1,63); row(i) = 1;
    part = repmat(row,15,1);
    targets = [targets; part];    
end
targets = targets';
outputs = prob_estimates_test';
%plotroc(targets,outputs);

[tpr,fpr,thresholds] = roc(targets,outputs);
temp_tpr = [];
for i=1:numberOfSubjects
    temp_tpr = [temp_tpr; tpr{i}];    
end
tpr_avg = mean(temp_tpr);
figure(); plot(fpr{1},tpr_avg);
grid on; hold on;
plot(fpr{1},1-fpr{1},'g');
legend('ROC curve', 'Diagonal Line');

%Find the intersection of y1 and y2 (This will give the EER)
x = fpr{1};
y1 = tpr_avg;
y2 = 1 - fpr{1};
[xout,yout] = intersections(x,y1,x,y2,1);
EER = (1 - yout) * 100;
hold on; plot(xout,yout,'r.','markersize',18);
legend('ROC curve', 'Diagonal Line', ['EER: ' num2str(EER) '%']);
%Print EER
disp(['EER: ' num2str(EER) '%']);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title('ROC');





