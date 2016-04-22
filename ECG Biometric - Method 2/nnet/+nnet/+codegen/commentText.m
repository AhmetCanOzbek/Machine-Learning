function text = commentText(text)
% Comment text

for i=1:numel(text)
  text{i} = ['% ' text{i}];
end
