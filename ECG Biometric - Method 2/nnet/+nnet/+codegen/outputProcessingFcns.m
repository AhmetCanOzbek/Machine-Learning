function fcns = outputProcessingFcns(net)

fcns = {};
for i=1:net.numLayers
  if net.outputConnect(i)
    fcns = [fcns net.outputs{i}.processFcns];
  end
end
fcns = unique(fcns);
