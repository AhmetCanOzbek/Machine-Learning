function code = moduleSubfunction(module,fcn,structName)

  import nnet.codegen.*;
  code = loadModuleFunction(module,fcn);
  code = stripCode(code);
  code{1} = strrep(code{1},fcn,[module '_' fcn]);
  code{1} = removeStructArgument(code{1});
  code = removeNoChangeBlockFromCode(code);
  [code,fields] = replaceStructFieldsInCode(code,structName);
  code{1} = addStructFieldArguments(code{1},structName,fields);
  code = addMissingEnd(code);
  code = [code(1) addMissingIndent(code(2:(end-1))) code(end)];
  
end

function code = stripCode(code)
% Remove blank and comment lines
  for j=numel(code):-1:1
    line = code{j};
    if isempty(line) || (line(1) == '%')
      code(j) = [];
    end
  end
end

function code = addMissingEnd(code)
  if isempty(strfind(code{end},'end'))
    code{end+1} = 'end';
  end
end

function code = addMissingIndent(code)
  for i=1:numel(code)
    if numel(code{i})<2 || any(code{i}(1:2) ~= '  ')
      code{i} = ['  ' code{i}];
    end
  end
end

function str = removeStructArgument(str)
% Remove the struct argument, which is the last argument
  parenPos = find(str == ')');
  commaPos = find(str==',',1,'last');
  str(commaPos:(parenPos-1)) = [];
end

function [code,fields] = replaceStructFieldsInCode(code,structName)
% Replace instances of "param.abc" with "param_abc",
% or "settings.abc" with "settings_abc"
  import nnet.codegen.*;
  oldRef = [structName '.'];
  newRef = [structName '_'];
  refLen = numel(oldRef);
  fields = {};
  for i=2:numel(code)
    str = code{i};
    pos = strfind(str,oldRef);
    for j=fliplr(pos+refLen)
      k = j;
      while (k<=numel(str)) && isFieldChar(str(k))
        k = k+1;
      end
      field = str(j:(k-1));
      if isempty(nnstring.match(field,fields))
        fields{end+1} = str(j:(k-1));
      end
    end
    for j=pos
      str(j-1+(1:refLen)) = newRef;
    end
    code{i} = str;
  end
end

function code = removeNoChangeBlockFromCode(code)
% Eliminate "if settings.no_change" block as generated
% code will not call functions if no_change is true
  i = 1;
  while (i<numel(code))
    if ~isempty(strfind(code{i},'if settings.no_change'))
      j = i+1;
      while (j <= numel(code)) && isempty(strfind(code{j},'end'))
        j = j+1;
      end
      code(i:j) = [];
    else
      i = i+1;
    end
  end
end

function str = addStructFieldArguments(str,structName,fields)
  import nnet.codegen.*;
  for i=1:numel(fields)
    fields{i} = [structName '_' fields{i}];
  end
  fields = sort(fields);
  commaPos = find(str==')',1,'last');
  str = [commaList([{str(1:(commaPos-1))} fields]) ')'];
end
