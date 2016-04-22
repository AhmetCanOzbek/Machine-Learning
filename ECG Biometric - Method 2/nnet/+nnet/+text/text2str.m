function str = text2str(text)

returnChar = sprintf('\n');
for i=1:numel(text)
  text{i}(end+1) = returnChar;
end
str = cat(2,text{:});