function perfs = apply(t,y,e,param)

% Copyright 2013 The MathWorks, Inc.


y = max(min(y,1),0);
t = max(min(t,1),0);
perfs = -(t.*log(max(y,eps)) + (1-t).*log(max(1-y,eps)));

