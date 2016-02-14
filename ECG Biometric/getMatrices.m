%R-320 , R-80

%Construct the matrices
%-80 hearbeats for training
%-15 heartbeats for testing
%-63 subjects

numberOfSubjects = 63;
numberOfTrainBeats = 80;
numberOfTestBeats = 15;

%%
%*Training
%Initialize feature matrix and label
train = zeros(63*80,400); %400 because R-320 , R-80
train_label = cell(63*80,1);
%Create 'train'
disp('Start constructing training matrix');
for i = 1:numberOfSubjects
    disp(['(ForTrain)Reading: Subject' num2str(i) ' FileName: ' fileNames{i,1}]);
    for j = 1:80        
        %Get the filtered ecg for each subject
        [qrs_amp_raw,qrs_i_raw,delay,filtered_ecg] = pan_tompkin(raw_ecg_subjects{i,1},Fs,0);
        filtered_ecg = filtered_ecg';
        %Crop R-320 , R-80 segment for each 80 heartbeat (Note: Starting
        %from the second R peak (second heartbeat))
        cropped = filtered_ecg(qrs_i_raw(j+2)-320 : qrs_i_raw(j+2)+79);
        %put it into train matrix
        train(80*(i-1) + j,:) = cropped;
    end    
end

%Create 'train_label'
label_names = fileNames(:,2);
for i = 1:numberOfSubjects
    for j = 1:80
        train_label{80*(i-1) + j,1} = label_names{i};
    end    
end
clear i; clear j;
disp('Train Matrix and Train Label Done.')

%%
%*Testing
%Initialize test matrix and label
test = zeros(63*15,400);
test_label = cell(63*15,1);
%Create 'test'
disp('Start constructing the test matrix');
for i = 1:numberOfSubjects
    disp(['(ForTest)Reading: Subject' num2str(i) ' FileName: ' fileNames{i,1}]);
    for j = 1:15        
        %Get the filtered ecg for each subject
        [qrs_amp_raw,qrs_i_raw,delay,filtered_ecg] = pan_tompkin(raw_ecg_subjects{i,1},Fs,0);
        filtered_ecg = filtered_ecg';
        %Crop R-320 , R-80 segment for each 15 heartbeat (Note: Starting
        %from the 84th R peak )
        cropped = filtered_ecg(qrs_i_raw(j+84)-320 : qrs_i_raw(j+84)+79);
        %put it into train matrix
        test(15*(i-1) + j,:) = cropped;
    end    
end
disp('Test Matrix Done.')

%Create 'test_label'
label_names = fileNames(:,2);
for i = 1:numberOfSubjects
    for j = 1:15
        test_label{15*(i-1) + j,1} = label_names{i};
    end    
end
clear i; clear j;
disp('Test Matrix and Test Label Done.')


