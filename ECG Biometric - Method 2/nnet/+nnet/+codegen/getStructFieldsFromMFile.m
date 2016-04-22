function fields = getStructFieldsFromMFile(module,fcn,structName)
% Get names of all "param" or "settings" fields used in function code

import nnet.codegen.*;

code = loadModuleFunction(module,fcn);
fields = getStructFieldsFromMCode(code,structName);
