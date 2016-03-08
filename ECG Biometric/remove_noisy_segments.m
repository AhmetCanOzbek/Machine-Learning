function[clean_ecg,window_number_ecg] = remove_noisy_segments(ecg,Fs,windowLength)
%Removes the noisy segments in an ecg signal explained in the paper:
%"Dry Contact Fingertip ECG-based Authentication System using Time, 
%Frequency Domain Features and Support Vector Machine"

%INPUTS: 
%ecg: input ecg signal
%Fs: sampling frequency of the input ecg signal
%windowLength: length of window segments to investigate to whether reject
%the window or not (in seconds)

%OUTPUTS:
%clean_ecg: Noise remove ecg signal

window_samples = Fs * windowLength;

%Subtract the mean
%ecg = ecg - mean(ecg);

window_number_ecg = zeros(1,length(ecg)); %variable to hold the window number in ecg
for i=1:length(ecg)      
    window_number_ecg(i) = floor(i/window_samples) + 1;   
end

%Calculate MCV for each window
n_of_windows = max(window_number_ecg); %N
mcv = zeros(1,n_of_windows);
for i = 1:n_of_windows
    segment = ecg(window_number_ecg == i);
    mcv(i) = sqrt(var(segment)) / (mean(segment))^2;
end

threshold = 2.5 * mean(mcv);
%Remove the windows with MCVs than higher than the threshold
for i=1:n_of_windows
    if mcv(i) > threshold
        %disp(['window: ' num2str(i)]);
        ecg(window_number_ecg == i) = nan;
    end
end

%Clear all NaNs
ecg(isnan(ecg)) = [];
clean_ecg = ecg;
end