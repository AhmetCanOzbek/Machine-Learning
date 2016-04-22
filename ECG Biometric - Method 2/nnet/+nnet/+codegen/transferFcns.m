function fcns = transferFcns(net)

fcns = {};
for i=1:net.numLayers
  fcns{end+1} = net.layers{i}.transferFcn;
end
fcns = unique(fcns);
