function [net, mse] = nn_train(X, Y, params)
% Fitting a two-layer feedforward neural network by stochastic gradient descent
% n =  # of data; d = # of features; h = # of hidden units
% Input:
%   - X (n X d): the input vector on the training data set
%   - Y (n X 1): the output vector on the training data set
%   - params: hyperparameters 
%        --params.hidden_units
%        --params.batch_size 
%        --params.step_size
%        --params.max_epoches
%
%
% Output:
%   - net: the parameters of your network
%      -- net.w (1 X h): two layer weights
%      -- net.V (h X d): first layer weights
%   - mse: the training error during ecach epoch

%% 
%initialize the parameter to some small random value     
net.V = unifrnd( -6/sqrt(params.hidden_units), 6/sqrt(params.hidden_units), [params.hidden_units, size(X, 2)]);  
net.w = unifrnd( -6/sqrt(params.hidden_units), 6/sqrt(params.hidden_units), [1, params.hidden_units]);

mse = [];
for iter = 1:params.max_epoches  
    for i = 1:params.batch_size:size(X, 1)
        % define the batch set        
        batch = i:min(i+params.batch_size-1, size(X, 1));
        Xbatch = X(batch, :); Ybatch = Y(batch);
        
        [~, grad] = nn_MSE(Xbatch, Ybatch, net);
        net.w = net.w - params.step_size * grad.w /size(Xbatch, 1);
        net.V = net.V - params.step_size * grad.V /size(Xbatch, 1);
    end
          
    % decrease the step size
    params.step_size = params.step_size / iter;    
    
    % keep track of the MSE across iterations
    mse = [mse, nn_MSE(X , Y, net)];      
    fprintf('nn_train: iter %d, mse %.4f\n', iter, mse(end));
       
    % stopping criterion
    if length(mse) > 1 && max(abs(mse(end) - mse(end-1))) < 1e-5
        break;
    end        
    
    
    %%%%%%%
    %Check Section 4 of this article for more suggestions: http://leon.bottou.org/publications/pdf/tricks-2012.pdf
    % http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/    
end
end


