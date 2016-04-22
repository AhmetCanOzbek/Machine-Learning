%Apply PCA
pca_dimension = 30;
disp(['Applyting PCA, PCA dim = ' num2str(pca_dimension)]);
%Crop analytic_component part
train_pca = train(:,1:end-numel(analytic_component));
test_pca = test(:,1:end-numel(analytic_component));
train_analytical_component = train(:,end-(numel(analytic_component)-1):end);
test_analytical_component = test(:,end-(numel(analytic_component)-1):end);
%Do PCA
[train_pca,mapping] = m_pca(train_pca,pca_dimension);
[test_pca,mapping] = m_pca(test_pca,pca_dimension,mapping);
%Merge back the analytic_component part
train = [train_pca,train_analytical_component];
test = [test_pca,test_analytical_component];