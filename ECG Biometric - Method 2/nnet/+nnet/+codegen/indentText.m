function text = indentText(text,n)
% Indent text

if nargin < 2
  n = 1;
end

spaces = repmat(' ',1,n*2);
for i=1:numel(text)
  text{i} = [spaces text{i}];
end
