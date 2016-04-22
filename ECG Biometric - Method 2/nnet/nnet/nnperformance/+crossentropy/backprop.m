function dy = backprop(t,y,e,param)

% Copyright 2013 The MathWorks, Inc.

y = max(min(y,1),0);
t = max(min(t,1),0);
dy = (-t./(y+eps)) + ((1-t)./(1-y+eps));
