function perf = crossentropy(varargin)
%CROSSENTROPY Cross-entropy performance function.
%
% <a href="matlab:doc crossentropy">crossentropy</a>(net,targets,outputs,perfWeights,...parameters...) calculates a
% network performance given targets, outputs, performance weights and parameters
% with a measure that heavily penalizes outputs which are extremely
% inaccurate (y near 1-t), with very little penalty for fairly
% correct classificatinos (y near t).  Minimizing cross-entropy leads to
% good classifiers.
%
% Only the first three arguments are required.  The default error weight
% is {1}, which weights the importance of all targets equally.
%
% The target values must be either 0 or 1.  The output values must fall in
% the interval [0, 1].
%
% Parameters are supplied as parameter name and value pairs:
%
% 'regularization' - a fraction between 0 (the default) and 1 indicating
%    the proportion of performance attributed to weight/bias values. The
%    larger this value the network will be penalized for large weights,
%    and the more likely the network function will avoid overfitting.
%
% 'normalization' - this can be 'none' (the default), or 'standard', which
%   results in outputs and targets being normalized to [-1, +1], and
%   therefore errors in the range [-2, +2), or 'percent' which normalizes
%   outputs and targets to [-0.5, 0.5] and errors to [-1, 1].
%
% Here a network's performance with 0.1 regularization is calculated.
%
%   perf = <a href="matlab:doc crossentropy">crossentropy</a>(net,targets,outputs,{1},'regularization',0.1)
%
% To setup a network to us the same performance measure during training:
%
%   net.<a href="matlab:doc nnproperty.net_performFcn">performFcn</a> = '<a href="matlab:doc crossentropy">crossentropy</a>';
%   net.<a href="matlab:doc nnproperty.net_performParam">performParam</a>.<a href="matlab:doc nnparam.regularization">regularization</a> = 0.1;
%   net.<a href="matlab:doc nnproperty.net_performParam">performParam</a>.<a href="matlab:doc nnparam.normalization">normalization</a> = 'none';
%
% See also MSE, MAE, SSE, SAE.

% Copyright 2013 The MathWorks, Inc.

% Function Info
persistent INFO;
if isempty(INFO), INFO = nnModuleInfo(mfilename); end
if (nargin == 1) && isrow(varargin{1}) && ischar(varargin{1})
  switch varargin{1}
    case 'info', perf = INFO;
    case 'defaultParam', perf = INFO.defaultParam;
  end
  return
end


% Arguments
if isa(varargin{1},'network') || isstruct(varargin{1})
  net = varargin{1};
  [args,net.performParam,nargs] = nnparam.extract_param(varargin,net.performParam);
  t = args{2};
  y = args{3};
  if (nargs < 4), ew = {1}; else ew = args{4}; end
  perf = nncalc.perform(net,t,y,ew);
else
  net.performFcn = mfilename;
  param = nn_modular_fcn.parameter_defaults(mfilename);
  [args,net.performParam,nargs] = nnparam.extract_param(varargin,param);
  t = args{1};
  y = args{2};
  if (nargs < 3), ew = {1}; end
  perf = nncalc.perform(net,t,y,ew);
end

