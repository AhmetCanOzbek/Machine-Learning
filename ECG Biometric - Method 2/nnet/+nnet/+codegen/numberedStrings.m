function list = numberedStrings(str,num)
% Generate numbered list of strings, such as 'x1', 'x2, etc.
  
list = cell(1,num);
for i=1:num
  list{i} = [str num2str(i)];
end
