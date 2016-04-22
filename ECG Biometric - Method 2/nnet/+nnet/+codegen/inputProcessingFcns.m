function fcns = inputProcessingFcns(net)

fcns = {};
for i=1:net.numInputs
  fcns = [fcns net.inputs{i}.processFcns];
end
fcns = unique(fcns);
  