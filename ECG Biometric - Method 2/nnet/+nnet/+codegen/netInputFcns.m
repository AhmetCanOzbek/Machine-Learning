function fcns = netInputFcns(net)

fcns = {};
for i=1:net.numLayers
  fcns{end+1} = net.layers{i}.netInputFcn;
end
fcns = unique(fcns);
