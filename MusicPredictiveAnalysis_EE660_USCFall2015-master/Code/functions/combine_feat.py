import numpy

def combine_feature(feature, confident=None):
    if confident is None:
        # find metrics of all features
        expanded_feat_mean = numpy.mean(feature, axis=0)
        expanded_feat_var = numpy.var(feature, axis=0)
        expanded_feat_std = numpy.std(feature, axis=0)
    else:
        # get only confident features
        confident_feature = feature[confident,:]    # = np.where(feature.where(confident)[0])
        # find metrics of confident features
        expanded_feat_mean = numpy.mean(confident_feature, axis=0)
        expanded_feat_var = numpy.var(confident_feature, axis=0)
        expanded_feat_std = numpy.std(confident_feature, axis=0)
    # concatenate to 36D feature space
    expanded_feat = numpy.concatenate((expanded_feat_mean, expanded_feat_var, expanded_feat_std))
    #return expanded feature space
    return expanded_feat

