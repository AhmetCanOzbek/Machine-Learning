function code = loadModuleFunction(module,fcn)

filename = ['+',module,filesep,fcn,'.m'];
code = nntext.load(filename)';