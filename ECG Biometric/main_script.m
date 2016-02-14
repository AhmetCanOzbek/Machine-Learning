clc; clear;
%Read files
getFiles; %gets 'fileNames' cell array into workspace
%Here I am just going to use just one session, so
fileNames = fileNames(1:63,:);


%%%%Read Input Signals for each subject%%%%
Fs = 1000;
bitRes = 12;
%Initialize
raw_ecg_subjects = cell(63,1);
h  = waitbar(0, 'Reading ECG files');
for i=1:63
    raw_ecg_subjects{i,1} = dlmread(fileNames{i,1},'\n',6,0); %6 because first six lines are headers in txt files
    waitbar(i/63,h);
end
delete(h); clear h;
disp('Files Read');

%Modify raw_ecg_subjects{20,1} (crop the beginning)
temp = raw_ecg_subjects{20,1};
raw_ecg_subjects{20,1} = temp(2500:end);


%R-320 , R-80
%Construct the matrices
%-80 hearbeats for training
%-15 heartbeats for testing
%-63 subjects
getMatrices;


%Save the matrices into file

