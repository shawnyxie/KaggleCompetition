%% load the training and testing dataset
clear all;
homedepot=load('trainData.txt'); % load the text file
homedepot_test=load('testData.txt');

rng(1);
Xtrain = homedepot(:,1:3); Ytrain = homedepot(:, 4); 
Xtest = homedepot_test(:, 1:3); 

% Feel free to tune these hyperparameters
% number of hidden units
params.hidden_units = 200;
% SGD, batch size
params.batch_size = 4;
% SGD, step size
params.step_size = 1e-3;
% epoches
params.max_epoches = 80;

% Try nn with 10 hidden units. 
params.hidden_units = 10; 
[net, mse] = nn_train( [ones(size(Xtrain,1),1),Xtrain], Ytrain, params);
figure;
plot(1:length(mse), mse, '-ro', 'LineWidth', 4, 'MarkerSize', 8);
xlabel('Epoch', 'FontSize', 20); ylabel('MSE', 'FontSize', 20);

% output the new predictions into an outputfile. 
nn_MSE_Output([ones(size(Xtest,1),1), Xtest], net);
%mse_train =  nn_MSE([ones(size(Xtrain,1),1), Xtrain], Ytrain, net);

%% use cross validation to select the best number of hidden units
%tic;
%hidden_units = [10 50 100 200];
% Set an initial number of folds for cross validation. 
%Nfolds = 10;
%n = size(Xtrain, 1);
%hidden_units_mse = zeros(1, size(hidden_units, 2));
%for i = 1:length(hidden_units)
%    params.hidden_units = hidden_units(i);
%    for j = 1:Nfolds
        %for each set of size 1/Nfold
%        test_set = (ceil((j-1)*n/Nfolds) + 1):(ceil(j*n/Nfolds));
%        [net, ~] = nn_train([ones(size(removerows(Xtrain, 'ind', test_set),1),1),removerows(Xtrain, 'ind', test_set)], removerows(Ytrain, 'ind', test_set), params);
%        mse_Vector(j) = nn_MSE([ones(size(Xtrain(test_set, :),1),1), Xtrain(test_set, :)], Ytrain(test_set, :), net);
%    end
%    hidden_units_mse(i) = mean(mse_Vector);
%end
%[~, index] = min(hidden_units_mse);
%hu_best = hidden_units(index);
%toc;
%disp(hu_best);
%mse_test = nn_MSE([ones(size(Xtest,1),1), Xtest], Ytest(), net); %total training error. 
%mse_train =  nn_MSE([ones(size(Xtrain,1),1), Xtrain], Ytrain(), net);
