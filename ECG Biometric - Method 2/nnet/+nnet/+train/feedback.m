function feedback(command,net,tr,options,varargin)

% Copyright 2007-2013 The MathWorks, Inc.

showWindow = net.trainParam.showWindow;
showCommandLine = net.trainParam.showCommandLine;
new_time = clock;

% No Java Compatibility
if ~usejava('swing')
  if (showWindow)
    showCommandLine = true;
    showWindow =  false;
  end
end

% Persistent
persistent NNTOOL_TIME;
persistent CHECKPOINT_TIME;
persistent CHECKPOINT_COUNT;
if isempty(NNTOOL_TIME)
  NNTOOL_TIME = [0 0 0 0 0 0];
end

% NNT 5.1 Backward Compatibility
if isnan(net.trainParam.show)
  showCommandLine = false;
end

switch command
  
  case 'start'
    
    algorithms = {net.divideFcn,net.trainFcn,net.performFcn,net.derivFcn};
    [status] = deal(varargin{:});
    if (showWindow)
      nntraintool('start',net,algorithms,status);
    end
    if (showCommandLine)
      disp(' ')
      disp(['Training with ' upper(net.trainFcn) '.']);
    end
        
  case 'update'
    
    if numel(varargin) == 6
      [data,calcLib,calcNet,bestNet,status,statusValues] = deal(varargin{:});
    else % numel(varargin) == 3
      [status,data,statusValues] = deal(varargin{:});
      calcLib = [];
      calcNet = [];
      bestNet = net;
    end
    
    doStart = (tr.num_epochs == 0);
    doStop = ~isempty(tr.stop);
    
    % Update NN Training Tool Window
    if (showWindow)
      nntraintool('clear_stops')
      if (doStart || doStop || (etime(new_time,NNTOOL_TIME) > 0.1))
        nntraintool('update',net,data,calcLib,calcNet,tr,statusValues);
        NNTOOL_TIME = new_time;
      end
    end
    
    % Checkpoints
    if ~isempty(options)
      [CHECKPOINT_TIME,CHECKPOINT_COUNT] = ...
        nnet.checkpoint.write(net,calcLib,bestNet,tr,options,CHECKPOINT_TIME,CHECKPOINT_COUNT);
    end
    
    % Update Command Line
    if (showCommandLine)
      if doStart || doStop || (rem(tr.num_epochs,net.trainParam.show)==0)
        numStatus = length(status);
        s = cell(1,numStatus*2-1);
        for i=1:length(status)
          s{i*2-1} = train_status_str(status(i),statusValues(i));
          if (i < numStatus), s{i*2} = ', '; end
        end
        disp([s{:}])
        if doStop
          disp(['Training with ' upper(net.trainFcn) ' completed: ' tr.stop])
          disp(' ');
        end
      end
    end
end

%%
function str = train_status_str(status,value)

if ~isfinite(status.max)
  str = [status.name ' ' num2str(value)];
else
  str = [status.name ' ' num2str(value) '/' num2str(status.max)];
end
