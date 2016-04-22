function[normalized_signal,max_signal, min_signal] = normalize(signal,varargin)
%Normalizes between 0 and 1

if nargin == 1
    max_signal = max(signal);
    min_signal = min(signal);   
end

if nargin == 3
    max_signal = varargin{1};
    min_signal = varargin{2};
end

normalized_signal = (signal - min_signal) / (max_signal - min_signal);

end