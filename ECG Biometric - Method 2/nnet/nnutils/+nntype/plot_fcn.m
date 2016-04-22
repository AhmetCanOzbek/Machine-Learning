function [out1,out2] = plot_fcn(in1,in2,in3)
%NN_PLOT_FCN Plot function type.

% Copyright 2010-2012 The MathWorks, Inc.

%% =======================================================
%  BOILERPLATE_START
%  This code is the same for all Type Functions.

  persistent INFO;
  if isempty(INFO), INFO = get_info; end
  if nargin < 1, error(message('nnet:Args:NotEnough')); end
  if ischar(in1)
    switch (in1)
      
      case 'info'
        % this('info')
        out1 = INFO;
        
      case 'isa'
        % this('isa',value)
        out1 = isempty(type_check(in2));
        
      case {'check','assert','test'}
        % [*err] = this('check',value,*name)
        nnassert.minargs(nargin,2);
        if nargout == 0
          err = type_check(in2);
        else
          try
            err = type_check(in2);
          catch me
            out1 = me.message;
            return;
          end
        end
        if isempty(err)
          if nargout>0,out1=''; end
          return;
        end
        if nargin>2, err = nnerr.value(err,in3); end
        if nargout==0, err = nnerr.value(err,'Value'); end
        if nargout > 0
          out1 = err;
        else
          throwAsCaller(MException(nnerr.tag('Type',2),err));
        end
        
      case 'format'
        % [x,*err] = this('format',x,*name)
        err = type_check(in2);
        if isempty(err)
          out1 = strict_format(in2);
          if nargout>1, out2=''; end
          return
        end
        out1 = in2;
        if nargin>2, err = nnerr.value(err,in3); end
        if nargout < 2, err = nnerr.value(err,'Value'); end
        if nargout>1
          out2 = err;
        else
          throwAsCaller(MException(nnerr.tag('Type',2),err));
        end
        
      case 'check_param'
        out1 = '';
      otherwise,
        try
          out1 = eval(['INFO.' in1]);
        catch me, nnerr.throw(['Unrecognized first argument: ''' in1 ''''])
        end
    end
  else
    error(message('nnet:Args:Unrec1'))
  end
end

%  BOILERPLATE_END
%% =======================================================

function info = get_info
  info = nnfcnFunctionType(mfilename,'Plot Function',7,...
    7,fullfile('nnet','nnplot'));
end

function err = type_check(x)
  err = nntest.fcn(x,false);
  if ~isempty(err), return; end
  info = feval(x,'info');
  if ~strcmp(info.type,'nntype.plot_fcn')
    err = ['VALUE info.type is not nntype.plot_fcn.'];
    return;
  end
  
  % Random stream
  saveRandStream = RandStream.getGlobalStream;
  RandStream.setGlobalStream(RandStream('mt19937ar','seed',pi));
  
  % Test that a training figure is created and no error happens
  %try
    inputs = 0:0.01:10;
    targets = sin(inputs);
    net = feedforwardnet(20);
    [net,data,tr] = nntraining.setup(net,net.trainFcn,{inputs},{},{},{targets},{1});
    tr.best_epoch = 0;
    tr.goal = 0;
    tr.states = {'epoch','time','perf','vperf','tperf','mu','gradient','val_fail'};
    perf = 1;
    gradient = 0.0001;
    mu = 1;
    status = ...
      [ ...
      nntraining.status('Epoch','iterations','linear','discrete',0,10,0), ...
      nntraining.status('Time','seconds','linear','discrete',0,100,0), ...
      nntraining.status('Performance','','log','continuous',perf,0.00001,perf) ...
      nntraining.status('Gradient','','log','continuous',gradient,1e-5,gradient) ...
      nntraining.status('Mu','','log','continuous',mu,1e10,mu) ...
      nntraining.status('Validation Checks','','linear','discrete',0,6,0) ...
      ];
    nnet.train.feedback('start',net,tr,[],status);
    tr = nnet.trainingRecord.update(tr,[0 0.5 perf 2 3 mu gradient 0]);
    
    param = feval(x,'defaultParam');
    f = feval(x,'training',net,tr,data,param);
    set(f,'visible','on')
    drawnow
    delete(f);
  %catch me
    %delete(get(0,'children'));
    %err = [upper(x) ' failed to plot.'];
  %end
  
  % Random Stream
  RandStream.setGlobalStream(saveRandStream);
end

function x = strict_format(x)
end
