function out1 = genFunction(net,filepath,varargin)
%GENFUNCTION Generate a MATLAB function for simulating a neural network.
%
% This function generates a complete stand-alone MATLAB function for
% simulating a neural network including all settings, weight and bias
% values, module functions and calculations in one file.
%
% Generating a MATLAB neural network simulation function helps:
%
% * Document the input-output transform of a neural network
% * Generate C/C++ code with <a href="matlab:doc coder">MATLAB Coder</a> <a href="matlab:doc codegen">codegen</a>
% * Generate efficient MEX functions with <a href="matlab:doc coder">MATLAB Coder</a> <a href="matlab:doc codegen">codegen</a>
% * Generate stand-alone C executables with <a href="matlab:doc compiler">MATLAB Compiler</a> <a href="matlab:doc mcc">mcc</a>
% * Generate C/C++ libraries with <a href="matlab:doc compiler">MATLAB Compiler</a> <a href="matlab:doc mcc">mcc</a>
% * Generate Excel and .COM components with <a href="matlab:doc matlabxl">MATLAB Builder EX</a> <a href="matlab:doc mcc">mcc</a> options
% * Generate Java components with <a href="matlab:doc javabuilder">MATLAB Builder JE</a> <a href="matlab:doc mcc">mcc</a> options
% * Generate .NET components with <a href="matlab:doc dotnetbuilder">MATLAB Builder NE</a> <a href="matlab:doc mcc">mcc</a> options
%
% genFunction(net,'path/name') takes a neural network and file path and
% produces a standalone MATLAB function file 'name.m'.
%
% genFunction(...,'MatrixOnly','yes') overrides the default cell/matrix
% notation and instead generates a function that only uses matrix
% arguments compatible with MATLAB Coder tools.  For static networks
% the matrix columns are interpreted as independent samples.  For
% dynamic networks the matrix columns are interpreted as a series of
% timesteps. The default value is 'no'.
%
% genFunction(...,'ShowLinks','no') disables the default behavior of
% displaying links to generated help and source code. The default is 'yes'.
%
% Three kinds of MATLAB functions can be generated:
%
% <strong>Static Neural Network</strong>
%
% Here a static network is trained and its outputs for the training
% data are calculated.
%
%   [x,t] = <a href="matlab:doc house_dataset">house_dataset</a>;
%   houseNet = <a href="matlab:doc feedforwardnet">feedforwardnet</a>(10);
%   houseNet = <a href="matlab:doc nnet/train">train</a>(houseNet,x,t);
%   y = houseNet(x);
% 
% A MATLAB function is generated and tested.  Then the new function is
% compiled to a shared/dynamically linked library with <a href="matlab:doc mcc">mcc</a>.
%
%   <a href="matlab:doc genFunction">genFunction</a>(houseNet,'houseFcn');
%   y2 = houseFcn(x);
%   accuracy2 = max(abs(y-y2))
%   <a href="matlab:doc mcc">mcc</a> -W lib:libHouse -T link:lib houseFcn
%
% Next another version of the MATLAB function is generated which only
% supports matrix arguments (no cell arrays).  This function is tested.
% Then it is used to generate a MEX function with the MATLAB Coder tool
% <a href="matlab:doc codegen">codegen</a> which is also tested.
%
%   <a href="matlab:doc genFunction">genFunction</a>(houseNet,'houseFcn','MatrixOnly','yes');
%   y3 = houseFcn(x);
%   accuracy3 = max(abs(y-y3))
%
%   x1Type = coder.typeof(double(0),[13 Inf]); % Coder type of input 1
%   <a href="matlab:doc codegen">codegen</a> houseFcn.m -config:mex -o houseCodeGen -args {x1Type}
%   y4 = houseCodeGen(x);
%   accuracy4 = max(abs(y-y4))
%   
% <strong>Dynamic Neural Network</strong>
%
% Here a dyanamic network is trained and its outputs for the training
% data are calculated.
%
%   [x,t] = <a href="matlab:doc maglev_dataset">maglev_dataset</a>;
%   maglevNet = <a href="matlab:doc narxnet">narxnet</a>(1:2,1:2,10);
%   [X,Xi,Ai,T] = <a href="matlab:doc preparets">preparets</a>(maglevNet,x,{},t);
%   maglevNet = <a href="matlab:doc nnet/train">train</a>(maglevNet,X,T,Xi,Ai);
%   [y,xf,af] = maglevNet(X,Xi,Ai);
%
% Next a MATLAB function is generated and tested. The function is then used
% to create a shared/dynamically linked library with <a href="matlab:doc mcc">mcc</a>.
%
%   <a href="matlab:doc genFunction">genFunction</a>(maglevNet,'maglevFcn');
%   [y2,xf,af] = maglevFcn(X,Xi,Ai);
%   accuracy2 = max(abs(cell2mat(y)-cell2mat(y2)))
%   <a href="matlab:doc mcc">mcc</a> -W lib:libMaglev -T link:lib maglevFcn
%
% Next another version of the MATLAB function is generated which only
% support matrix arguments (no cell arrays).  This function is tested.
% Then it is used to generate a MEX function with the MATLAB Coder tool
% <a href="matlab:doc codegen">codegen</a> which is also tested.
%
%   genFunction(maglevNet,'maglevFcn','MatrixOnly','yes');
%   x1 = cell2mat(X(1,:)); % Convert each input to matrix
%   x2 = cell2mat(X(2,:));
%   xi1 = cell2mat(Xi(1,:)); % Convert each input state to matrix
%   xi2 = cell2mat(Xi(2,:));
%   [y3,xf1,xf2] = maglevFcn(x1,x2,xi1,xi2);
%   accuracy3 = max(abs(cell2mat(y)-y3))
%
%   x1Type = coder.typeof(double(0),[1 Inf]); % Coder type of input 1
%   x2Type = coder.typeof(double(0),[1 Inf]); % Coder type of input 2
%   xi1Type = coder.typeof(double(0),[1 2]); % Coder type of input 1 states
%   xi2Type = coder.typeof(double(0),[1 2]); % Coder type of input 2 states
%   <a href="matlab:doc codegen">codegen</a> maglevFcn.m -config:mex -o maglevNetCodeGen -args {x1Type x2Type xi1Type xi2Type}
%   [y4,xf1,xf2] = maglevNetCodeGen(x1,x2,xi1,xi2);
%   dynamic_codegen_accuracy = max(abs(cell2mat(y)-y4))
%
% See also, GENSIM.

  import nnet.codegen.*;
  import nnet.arguments.*;

  % Path and Name
  if nargin < 2, filepath = 'neural_function'; end
  [filepath,name] = fileparts(filepath);
  
  % Options
  options.MatrixOnly = 'no';
  options.ShowLinks = 'yes';
  options = nnet.options.override(options,varargin);
  MatrixOnly = options.MatrixOnly;
  if ~ischar(MatrixOnly) || ~isrow(MatrixOnly) || isempty(nnstring.match(MatrixOnly,{'yes','no'}))
    error('nnet:genFunction:args','Option ''MatrixOnly'' must be set to ''no'' (default) or ''yes''.');
  end
  MatrixOnly = strcmp(MatrixOnly,'yes');
  ShowLinks = options.ShowLinks;
  if ~ischar(ShowLinks) || ~isrow(ShowLinks) || isempty(nnstring.match(ShowLinks,{'yes','no'}))
    error('nnet:genFunction:args','Option ''ShowLinks'' must be set to ''no'' (default) or ''yes''.');
  end
  ShowLinks = strcmp(ShowLinks,'yes');
  
  % Simplify Network
  [net,pi,pl,po] = prune(net);
  if ~isempty(pi)
    disp('Notice: Zero sized inputs were removed.');
  end
  if ~isempty(pl)
    disp('Notice: Zero sized layers were removed.');
  end
  if ~isempty(po)
    disp('Notice: Zero sized outputs were removed.');
  end
  net = struct(net);
  net = weedProcessSteps(net);
  
  % Generate code
  if ~MatrixOnly
    code = genFunctionCell(net,name,filepath);
  elseif ((net.numInputDelays + net.numLayerDelays) == 0)
    code = genFunctionMatrixStatic(net,name,filepath);
  else
    code = genFunctionMatrixDynamic(net,name,filepath);
  end
  
  % Return code if there is an output argument
  if (nargout >= 1)
    out1 = code;
    return
  end
  
  % Save
  clear(name)
  fullpath = fullfile(filepath,[name '.m']);
  nntext.save(fullpath,code);
  path(path);
  
  % Feedback
  if ShowLinks
    disp(' ')
    disp(['MATLAB function generated: ' fullpath])
    disp(['To view generated function code: <a href="matlab:edit ' name '">edit ' name '</a>'])
    disp(['For examples of using function: <a href="matlab:help ' name '">help ' name '</a>'])
    disp(' ')
  end
end

