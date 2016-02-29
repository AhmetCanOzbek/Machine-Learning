function[filtered_ecg] = filter_ecg(ecg, Fs, filterOrder, h_cutoff, l_cutoff)
%Filters the ecg signal
%INPUTS:
%ecg: Input ECG signal
%Fs: Sampling frequency
%filterOrder: Order of the filter
%h_cutoff: High frequency cutoff of the bandpass filter (Hz)
%l_cutoff: Low frequency cutoff of the bandpass filter (Hz)
% 
%OUTPUTS:
%filtered_ecg: Filtered ecg signal


%Normalize
ecg = ecg - mean(ecg);

%Design the FIR filter
b = fir1(filterOrder,[(2/Fs)*(h_cutoff) (2/Fs)*(l_cutoff)]);

%Zero pad the end
delay = round(mean(grpdelay(b,length(ecg),Fs)));
ecg = [ecg; zeros(delay,1)];

%Filter
filtered_ecg = filter(b,1,ecg);
%Compansate for the delay caused by filtering the signal
filtered_ecg(1:delay) = [];

end