clc; clear;
load subjects_raw_data.mat
disp('*****Benchmark script start*****');

%Construct the matrices
%-80 hearbeats for training
%-15 heartbeats for testing
%-63 subjects
numberOfSubjects = 63;
numberOfTrainBeatsVector = 80:-5:5;
numberOfTestBeats = 15;

getFiles;

%%
%Allocate variables for results
train_accuracy_result = zeros(length(numberOfTrainBeatsVector),1);
test_accuracy_result = zeros(length(numberOfTrainBeatsVector),1);
eer_result = zeros(length(numberOfTrainBeatsVector),1);

for c=1:length(numberOfTrainBeatsVector)
    
    progressbar(c/length(numberOfTrainBeatsVector));
    numberOfTrainBeats = numberOfTrainBeatsVector(c);
    
    Fs = 1000;
    filterOrder = 75;
    h_cutoff = 0.45; %Hz (HighPass Cutoff frequency of bandbass filter)
    l_cutoff = 35; %Hz (LowPass Cutoff frequency of bandbass filter)
    m = -275; %m samples before R peak
    n = 265; %n samples after R peak
    F_m = -80; %m samples before R peak (for FFT vector)
    F_n = +80; %n samples after R peak (for FFT vector)
    n_of_fft_samples = 45;
    %Initialize feature matrix and label
    n_of_features = ( abs(m-n)+1 + n_of_fft_samples ); %Dimensionality
    %Allocate
    train = zeros(numberOfSubjects*numberOfTrainBeats,n_of_features);
    train_label = cell(numberOfSubjects*numberOfTrainBeats,1);
    test = zeros(numberOfSubjects*numberOfTestBeats,n_of_features);
    test_label = cell(numberOfSubjects*numberOfTestBeats,1);
    
    for i = 1:numberOfSubjects
        %disp(['Reading: Subject' num2str(i) ' FileName: ' fileNames{i,1}]);
        ecg = raw_ecg_subjects{i,1};
        %Remove noisy segments
        ecg = remove_noisy_segments(ecg,1000,1);
        %Filter
        filtered_ecg = filter_ecg(ecg, Fs, filterOrder, h_cutoff, l_cutoff);
        %Get R peaks
        [qrs_amp_raw,qrs_i_raw,delay,~] = pan_tompkin(filtered_ecg,Fs,0);
        filtered_ecg = filtered_ecg';
        
        %TRAIN Matrix
        for j = 1:numberOfTrainBeats
            %Temporal Component: Crop R-m , R+n segment for each 80 heartbeat
            %(Note: Starting from the second R peak (second heartbeat))
            temporal_component = filtered_ecg(qrs_i_raw(j+2)+(m) : qrs_i_raw(j+2)+(n));
            %Frequency Component (R-F_m, R+F_n)
            frequency_component = abs(fft(filtered_ecg(qrs_i_raw(j+2)+(F_m) : qrs_i_raw(j+2)+(F_n)),n_of_fft_samples));
            %put it into train matrix (cascade temporal component and frequency component)
            train(numberOfTrainBeats*(i-1) + j,:) = [temporal_component frequency_component];
        end
        
        %TEST Matrix
        for j = 1:numberOfTestBeats
            %Temporal Component: Crop R-m , R+n segment for each 80 heartbeat
            %(Note: Starting from the second R peak (second heartbeat))
            temporal_component = filtered_ecg(qrs_i_raw(j+84)+(m) : qrs_i_raw(j+84)+(n));
            %Frequency Component (R-F_m, R+F_n)
            frequency_component = abs(fft(filtered_ecg(qrs_i_raw(j+84)+(F_m) : qrs_i_raw(j+84)+(F_n)),n_of_fft_samples));
            %put it into train matrix (cascade temporal component and frequency component)
            test(numberOfTestBeats*(i-1) + j,:) = [temporal_component frequency_component];
        end
    end
    
    %Create 'train_label'
    label_names = fileNames(:,2);
    for i = 1:numberOfSubjects
        for j = 1:numberOfTrainBeats
            train_label{numberOfTrainBeats*(i-1) + j,1} = label_names{i};
        end
    end
    clear i; clear j;
    disp('Train Matrix and Train Label Done.')
    
    
    %Create 'test_label'
    label_names = fileNames(:,2);
    for i = 1:numberOfSubjects
        for j = 1:numberOfTestBeats
            test_label{numberOfTestBeats*(i-1) + j,1} = label_names{i};
        end
    end
    clear i; clear j;
    disp('Test Matrix and Test Label Done.');
    
    %Normalize Train and Test
    %Iterate each column for the train and test
    for j=1:size(train,2)
        %disp(['Normalizing Column: ' num2str(j)]);
        %Normalize train
        [train(:,j),max_train,min_train] = normalize(train(:,j));
        %Normalize test
        [test(:,j),~,~] = normalize(test(:,j),max_train,min_train);
    end
    
    %Save the train, train_label, test, test_label
    save('train_test.mat','train','train_label','test','test_label');
    
    disp(['Number of Subjects: ' num2str(numberOfSubjects)]);
    disp(['Number of Train Beats: ' num2str(numberOfTrainBeats)]);
    disp(['Number of Test Beats: ' num2str(numberOfTestBeats)]);
    disp('Parameters:');
    disp(['*Fs = ' num2str(Fs) ' (Hz)']);
    disp(['*filterOrder = ' num2str(filterOrder)]);
    disp(['*h_cutoff = ' num2str(h_cutoff) ' (Hz)']);
    disp(['*l_cutoff = ' num2str(l_cutoff) ' (Hz)']);
    disp(['*m = ' num2str(m) ' (samples)']);
    disp(['*n = ' num2str(n) ' (samples)']);
    disp(['*F_m = ' num2str(F_m) ' (samples)']);
    disp(['*F_n = ' num2str(F_n) ' (samples)']);
    disp(['*n_of_fft_samples = ' num2str(n_of_fft_samples) ' (samples)']);
    disp('Results:');
    
    %%
    classification;
    
    %%
    eer;
    
    
    %%
    %Save the results
    train_accuracy_result(c,1) = accuracy_train(1);
    test_accuracy_result(c,1) = accuracy_test(1);
    eer_result(c,1) = EER;   
    
end

%Plotting



disp('Done.');
disp(' ');