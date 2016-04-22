function [calcMode,err] = options2Mode(net,nameValuePairs)

% Copyright 2013 The MathWorks, Inc.

calcMode = [];

%=================== Override Default Options

options = nnet.options.calc.defaults;
options.reduction = net.efficiency.memoryReduction;
[options,err] = nnet.options.override(options,nameValuePairs);
if ~isempty(err), return; end
err = nnet.options.calc.check(options);
if ~isempty(err), return, end

% Checkpoint File Expand
options.CheckpointFile = nnet.checkpoint.expandFile(options.CheckpointFile);

%=================== Set Calculation Mode

% Pick the calculation mode
if isdeployed
  calcMode = nnMATLAB;
elseif strcmp(options.useParallel,'yes') && ~strcmp(options.useGPU,'no')
  calcMode = nnParallel('subcalc',nnGPUOp('precision',options.precision),...
    'onlyGPUs',strcmp(options.useGPU,'only'),'direction',options.direction);
elseif strcmp(options.useParallel,'yes')
  calcMode = nnParallel('subcalc',MexOrMATLAB(net,options,true),'direction',options.direction);
elseif ~strcmp(options.useGPU,'no')
  calcMode = nnGPUOp('precision',options.precision);
else
  calcMode = MexOrMATLAB(net,options,false);
end

%=================== Set Other Options

calcMode.options = options;

%===================

function calcMode = MexOrMATLAB(net,options,isParallel)
calcMode = nncalc.defaultMode(net,[],isParallel);
calcMode.hints.direction = options.direction;
if (options.reduction > 1) && ~strcmp(calcMode.mode,'nnMex')
  calcMode = nnMemReduc('reduction',options.reduction,'subcalc',calcMode);
end

