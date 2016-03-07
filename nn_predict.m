function hatY = nn_predict(X , net)
% Making prediction for two-layer feedforward neural network
% n = # of data points; d = # of data features; h = # of hidden units
% Input:
%   - X (n X d): the feature
%   - net: a structure that stores the network weights
%      -- net.w (1 X h): top layer weights
%      -- net.V (h X d): first layer weights

% Output:
%   - hatY (n X 1): the prediction 

    

hatY  = (net.w * sigmoid(net.V * X'))'; % prediction

return;
end


function f  = sigmoid(x)
% sigmoid function: 
    f = 1 ./ (1 + exp(-x));
end
