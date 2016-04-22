function dy = backpropReverse(dx,x,y,settings)
%MAPMINMAX.BACKPROPREVERSE

% Copyright 2012 The MathWorks, Inc.

dy = bsxfun(@rdivide,dx,settings.gain);
