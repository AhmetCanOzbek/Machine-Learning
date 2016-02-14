files = dir('/Users/ahmetcanozbek/Desktop/EE590/ecgProject/CYBHi_data/long-term');

%Start at 4 to ignore '.', '..' (things like that)
for i=4:size(files,1)
    fileNames{i-3,1} = files(i).name;   
    temp_split = strsplit(fileNames{i-3,1},'-'); %To get the subject name
    fileNames{i-3,2} = temp_split{2}; %To get the subject name
end

clear files
clear i
clear temp_split