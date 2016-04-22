function str = operatorList(list,op)
% Combine strings with operators between them

if isempty(list)
  str = '';
else
  str = list{1};
  for i=2:numel(list)
    str = [str op list{i}];
  end
end
