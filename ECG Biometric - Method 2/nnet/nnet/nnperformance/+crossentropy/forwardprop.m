function dperf = forwardprop(dy,t,y,e,param)

% Copyright 2013 The MathWorks, Inc.

dy = (-t./(y+eps)) + ((1-t)./(1-y+eps));
dperf = bsxfun(@times,dy,dy);
