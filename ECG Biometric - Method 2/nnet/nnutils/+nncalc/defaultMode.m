function calcMode = defaultMode(net,calcMode,isParallel)

% Copyright 2012 The MathWorks, Inc.

if isdeployed
  calcMode = nnMATLAB;
  
elseif (nargin < 2) || isempty(calcMode)
  
  if ~isdeployed && ~isempty(net.trainFcn)
    trainInfo = feval(net.trainFcn,'info');
    usesGradient = trainInfo.usesGradient;
    usesJacobian = trainInfo.usesJacobian;
  else
    usesGradient = false;
    usesJacobian = false;
  end
  
  % Default is MEX2 or nn7, unless custom functions are used
  if usesJacobian && (net.numLayerDelays == 0)
    calcMode = nn7;
  else
    calcMode = nnMex2;
  end
  if ~isempty(calcMode.netCheck(net,calcMode.hints,usesGradient,usesJacobian))
    calcMode = nnMATLAB;
  end

elseif isfield(calcMode.hints,'subcalc')
  if strcmp(calcMode.hints.subcalc.name,'default')
    calcMode.hints.subcalc = nncalc.defaultMode(net);
  end
  calcMode.hints.subcalc = nncalc.defaultMode(net,calcMode.hints.subcalc);

elseif isfield(calcMode.hints,'subcalcs')
  for i=1:numel(calcMode.hints.subcalcs)
    calcMode.hints.subcalcs{i} = nncalc.defaultMode(net,calcMode.hints.subcalcs{i});
  end
end
