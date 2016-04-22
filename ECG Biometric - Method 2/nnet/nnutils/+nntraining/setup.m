function [net,data,tr,err] = setup(net,trainFcn,X,Xi,Ai,T,EW,configNetEnable)

% Copyright 2010-2012 The MathWorks, Inc.

if nargin < 8, configNetEnable = true; end
data = struct;
tr = [];
err = [];

net = feval(net.trainFcn,'formatNet',net);

if ~isa(X,'gpuArray')
  X = nntype.data('format',X,'Inputs X');
  T = nntype.data('format',T,'Targets T');
  Xi = nntype.data('format',Xi,'Input states Xi');
  Ai = nntype.data('format',Ai,'Layer states Ai');
  EW = nntype.nndata_pos('format',EW,'Error weights EW');

  % Configure network inputs and outputs
  [net,X,Xi,Ai,T,EW,Q,TS,err] = nntraining.config(net,X,Xi,Ai,T,EW,configNetEnable);
  if ~isempty(err)
    return;
  end
  net = struct(net);

  % If any NaN values in inputs:
  % 1) Calculate outputs
  % 2) For each NaN output, set associated T to NaN
  % 3) Fill in NaN inputs with 0
  T = nntraining.fix_nan_inputs(net,X,Xi,Ai,T,Q,TS);
  data.format = 'CELLofMATRIX';
  
else
  
  % NNGPUOP
  err = setup_nngpuop(net,X,Xi,Ai,T,EW);
  if isempty(err)
    [err,net,X,Xi,Ai,T,EW,Q,TS] = setup_nngpuop(net,X,Xi,Ai,T,EW);
    data.format = 'CELLofGPU';
  
  % NNGPU
  else
    [err2,net,X,Xi,Ai,T,EW,Q,TS] = setup_nngpu(net,X,Xi,Ai,T,EW);
    if ~isempty(err2)
      return
    else
      err = '';
    end
    data.format = 'NNDATA2GPU';
  end
end

% Data Structure
data.X = X;
data.Xi = Xi;
data.Ai = Ai;
data.T = T;
data.EW = EW;
data.Q = Q;
data.TS = TS;

% Divide Data
outputN = nn.output_sizes(net);
if ~isempty(net.divideFcn)
  divideFcn = net.divideFcn;
  switch net.divideMode
    case 'none'
      trainInd = 1:Q;
      valInd = [];
      testInd = [];
      data.train = all_data(data,'Training',outputN,trainInd);
      data.val = disabled_data(data,'Validation',outputN);
      data.test = disabled_data(data,'Test',outputN);
    case 'sample'
      [trainInd,valInd,testInd] = feval(net.divideFcn,Q,net.divideParam);
      data.train = share_samples(data,trainInd,'Training',outputN);
      data.val = share_samples(data,valInd,'Validation',outputN);
      data.test = share_samples(data,testInd,'Test',outputN);
    case 'time',
      [trainInd,valInd,testInd] = feval(net.divideFcn,TS,net.divideParam);
      data.train = share_timesteps(data,trainInd,'Training',outputN);
      data.val = share_timesteps(data,valInd,'Validation',outputN);
      data.test = share_timesteps(data,testInd,'Test',outputN);
    case 'sampletime',
      Q_TS = Q * TS;
      [trainInd,valInd,testInd] = feval(net.divideFcn,Q_TS,net.divideParam);
      data.train = share_sampleTimesteps(data,trainInd,'Training',outputN);
      data.val = share_sampleTimesteps(data,valInd,'Validation',outputN);
      data.test = share_sampleTimesteps(data,testInd,'Test',outputN);
    case 'value',
      N_Q_TS = sum(outputN)*Q*TS;
      [trainInd,valInd,testInd] = feval(net.divideFcn,N_Q_TS,net.divideParam);
      data.train = share_general(data,trainInd,'Training',outputN);
      data.val = share_general(data,valInd,'Validation',outputN);
      data.test = share_general(data,testInd,'Test',outputN);
  end
else
  divideFcn = 'dividetrain';
  trainInd = 1:Q;
  valInd = [];
  testInd = [];
  data.train = all_data(data,'Training',outputN,[]);
  data.val = disabled_data(data,'Validation',outputN);
  data.test = disabled_data(data,'Test',outputN);
end
trainInfo = feval(trainFcn,'info');
if ~trainInfo.usesValidation
  trainInd = union(trainInd,valInd);
  data.train.indices = trainInd;
  for i=1:numel(data.train.mask)
    data.train.mask{i}(~isnan(data.val.mask{i})) = 1;
    data.val.mask{i}(:) = NaN;
  end
  valInd = [];
  data.val.enabled = false;
  data.val.indices = valInd;
end

% Training record
tr = nnetTrainingRecord(net);
tr.divideFcn = divideFcn;
tr.divideMode = net.divideMode;
tr.trainInd = trainInd;
tr.valInd = valInd;
tr.testInd = testInd;
tr.trainMask = data.train.mask;
tr.valMask = data.val.mask;
tr.testMask = data.test.mask;

% ====================================================================

function [err,net,X,Xi,Ai,T,EW,Q,TS] = setup_nngpuop(net,X,Xi,Ai,T,EW)
[Q,TS] = deal([]);

err = '';
if (net.numInputDelays + net.numLayerDelays) > 0
  err = 'Cannot accept gpuArray data for dynamic neural networks.';
  return
end
if (net.numInputs ~= 1) || (net.numOutputs ~= 1)
  err = 'Can only accept gpuArray data for single input, single output networks.';
  return
end

% Q
if ~isempty(X)
  Q = size(X,2);
elseif ~isempty(T)
  Q = size(T,2);
else
  Q = 0;
end

% TS
TS = 1;

% Network dimensions
Ni = net.inputs{1}.size;
No = nn.output_sizes(net);

% Expand empty values
if isempty(X), X = {gpuArray(nan(Ni,Q))}; end
if isempty(Xi), Xi = cell(1,0); end
if isempty(Ai), Ai = cell(net.numLayers,0); end
if isempty(T), T = {gpuArray(nan(No,Q))}; end
if isempty(EW), EW = {gpuArray(ones(1,1))}; end
if ~iscell(X), X = {X}; end
if ~iscell(T), T = {T}; end
if ~iscell(EW), EW = {EW}; end

% Check X
if any(size(X) ~= [1 TS])
  err = 'Incorrectly sized inputs X.';
  return
end
for i=1:numel(X)
  x = X{i};
  if size(x,1) ~= Ni
    if (Ni == 0)
      err = 'Network must be configured with CONFIGURE before training with gpuArray data.';
    else
      err = 'Incorrectly sized inputs X.';
      return
    end
  end
  if size(x,2) ~= Q
    err = 'Incorrectly sized inputs X.';
    return
  end
end

% Check Xi
if ~isempty(Xi)
  err = 'Incorrectly sized input states Xi.';
  return
end

% Check Ai
if ~isempty(Ai)
  err = 'Incorrectly sized layer states Ai.';
  return
end

% Check T
if any(size(T) ~= [1 TS])
  err = 'Incorrectly sized targets T.';
  return
end
for i=1:numel(T)
  t = T{i};
  if size(t,1) ~= No
    if (No == 0)
      err = 'Network must be configured with CONFIGURE before training with gpuArray data.';
    else
      err = 'Incorrectly sized targets T.';
      return
    end
  end
  if size(x,2) ~= Q
    err = 'Incorrectly sized targets T.';
    return
  end
end

% Check EW
if (size(EW,1) ~= 1) || ((size(EW,2) ~= TS) && (size(EW,2) ~= 1))
  err = 'Incorrectly sized error weights EW.';
  return
end
for i=1:numel(EW)
  ew = EW{i};
  if (size(ew,1) ~= No) && (size(ew,1) ~= 1)
    if (No == 0)
      err = 'Network must be configured with CONFIGURE before training with gpuArray data.';
    else
      err = 'Incorrectly sized error weights EW.';
      return
    end
  end
  if (size(x,2) ~= Q) && (size(x,2) ~= 1)
    err = 'Incorrectly sized error weights EW.';
    return
  end
end



% ====================================================================

function [err,net,X,Xi,Ai,T,EW,Q,TS] = setup_nngpu(net,X,Xi,Ai,T,EW)
err = '';
[Q,TS] = deal([]);

% Precision
precision = classUnderlying(X);
  
% QQ
QQs = [size(X,1) size(Xi,1) size(Ai,1) size(T,1)];
QQs(QQs == 0) = [];
QQ = max([0 QQs]);
if any(QQs ~= QQ)
  err = 'Number of samples (rows of gpuArrays) of data arguments do not match.';
  return
end

% Q
if ~isempty(T)
  Qv = T;
elseif ~isempty(X)
  Qv = X;
elseif ~isempty(Xi)
  Qv = Xi;
elseif ~isempty(Ai)
  Qv = Ai;
else
  Qv = [];
end
realRows = gather(any(isfinite(Qv),2));
Q = find(realRows,1,'last');

% Network dimensions
Ni = sum(nn.input_sizes(net));
No = sum(nn.output_sizes(net));
Nl = sum(nn.layer_sizes(net));
NID = net.numInputDelays;
NLD = net.numLayerDelays;
anyInputsZero = any(nn.input_sizes(net)==0);
anyOutputsZero = any(nn.output_sizes(net)==0);

% Infer TS
Ni_TS = size(X,2);
No_TS = size(T,2);
if (Ni_TS == 0) && (No_TS == 0)
  TS = 0;
elseif (Ni > 0)
  TS = Ni_TS / Ni;
  if (TS ~= floor(TS))
    if anyInputsZero
      err = 'Input data size  (gpuArray columns) does not match input sizes. Fix data or CONFIGURE network.';
    else
      err = 'Input data size  (gpuArray columns) does not match input sizes.';
    end
    return;
  end
elseif (No > 0)
  TS = No_TS / No;
  if (TS ~= floor(TS))
    if anyOutputsZero
      err = 'Target data size (gpuArray columns) does not match output sizes. Fix data or CONFIGURE network.';
    else
      err = 'Target data size (gpuArray columns) does not match output sizes.';
    end
    return;
  end
else
  TS = 0;
end

% Expand empty values
if isempty(X), X = gpuArray(nan(QQ,Ni*TS,precision)); end
if isempty(Xi), Xi = gpuArray(nan(QQ,Ni*NID,precision)); end
if isempty(Ai), Ai = gpuArray(nan(QQ,Nl*NLD,precision)); end
if isempty(T), T = gpuArray(nan(QQ,No*TS,precision)); end
if isempty(EW), EW = gpuArray(ones(1,1,precision)); end

% Check sizes
if any(size(X) ~= [QQ Ni*TS])
  if anyInputsZero
    err = 'Input data size  (gpuArray columns) does not match input sizes. Fix data or CONFIGURE network.';
  else
    err = 'Input data size  (gpuArray columns) does not match input sizes.';
  end
  return
end
if any(size(Xi) ~= [QQ Ni*NID])
  err = 'Input state size  (gpuArray columns) does not match input sizes times input delay states.';
end
if any(size(Ai) ~= [QQ Nl*NLD])
  err = 'Layer state size  (gpuArray columns) does not match layers sizes times layer delay states.';
end
if any(size(T) ~= [QQ No*TS])
  if anyOutputsZero
    err = 'Target data size  (gpuArray columns) does not match output sizes. Fix data or CONFIGURE network.';
  else
    err = 'Target data size  (gpuArray columns) does not match output sizes.';
  end
end
if (size(EW,1) ~= 1) && (size(EW,1) ~= QQ)
  err = 'X and EW have different numbers of samples (gpuArray rows).';
  return
end
EWcols = size(EW,2);
allowed1 = (EWcols == 1);
allowed2 = (EWcols == TS);
allowed3 = (EWcols == TS*No);
if ~(allowed1 || allowed2 || allowed3)
  error('To avoid ambiguity gpuArray EW must have 1, TS or TS*(number of output elements) columns.');
end
  
% ====================================================================

function y = all_data(data,name,outputN,indices)
mask = ones(sum(outputN),data.Q*data.TS);
y.name = name;
y.enabled = true;
y.all = true;
y.masked = false; % TODO - remove this
y.indices = indices;
y.mask = mat2cell(mask,outputN,data.Q*ones(1,data.TS));

function y = share_samples(data,indices,name,outputN)
if isempty(indices)
  y = disabled_data(data,name,outputN); return;
elseif length(indices) == data.Q
  y = all_data(data,name,outputN,indices); return;
end
mask = nan(sum(outputN),data.Q*data.TS);
[ts,q] = meshgrid(1:data.TS,indices);
indices2 = (ts(:)-1)*data.Q+q(:);
mask(:,indices2) = 1;
y.name = name;
y.enabled = true;
y.all = false;
y.indices = indices;
y.masked = true;
y.mask = mat2cell(mask,outputN,data.Q*ones(1,data.TS));

function y = share_timesteps(data,indices,name,outputN)
if isempty(indices)
  y = disabled_data(data,name,outputN); return;
end
mask = NaN(sum(outputN),data.Q*data.TS);
[ts,q] = meshgrid(indices,1:data.Q);
indices2 = (ts(:)-1)*data.Q+q(:);
mask(:,indices2) = 1;
y.name = name;
y.enabled = true;
y.all = false;
y.masked = true;
y.indices = indices;
y.mask = mat2cell(mask,outputN,data.Q*ones(1,data.TS));

function y = share_sampleTimesteps(data,indices,name,outputN)
if isempty(indices)
  y = disabled_data(data,name,outputN); return;
end
mask = NaN(sum(outputN),data.Q*data.TS);
mask(:,indices) = 1;
y.name = name;
y.enabled = true;
y.all = false;
y.masked = true;
y.indices = indices;
y.mask = mat2cell(mask,outputN,data.Q*ones(1,data.TS));

function y = share_general(data,indices,name,outputN)
if isempty(indices)
  y = disabled_data(data,name,outputN); return;
end
mask = NaN(sum(outputN),data.Q*data.TS);
mask(indices) = 1;
y.name = name;
y.enabled = true;
y.all = false;
y.masked = true;
y.indices = indices;
y.mask = mat2cell(mask,outputN,data.Q*ones(1,data.TS));

function y = disabled_data(data,name,outputN)
mask = NaN(sum(outputN),data.Q*data.TS);
y.name = name;
y.enabled = false;
y.all = false;
y.masked = false;
y.indices = [];
y.mask = mat2cell(mask,outputN,data.Q*ones(1,data.TS));

