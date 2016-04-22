function s = poolSize

% Copyright 2012 The MathWorks, Inc.

if ~nnDependency.distCompAvailable
  s = 0;
else
  try
    if exist('parpool','file')
      pool = gcp;
      if isempty(pool)
        s = 0;
      else
        s = pool.NumWorkers;
      end
    else
      % TODO - Remove after R2013b release
      s = matlabpool('size');
    end
  catch
    % MATLABPOOL may fail if Java is not available (-nojvm, etc)
    s = 0;
  end
end
