function [W, b, J_plot] = trainNN(X, Y, h, N_h, LR, E)
% This function is used to train a neural network of a specified set of
% topology and hyperparameters.
%
% Inputs:
% X - input features
% Y - input labels
% h - number of hidden layers
% N_h - number of neurons/nodes per hidden layers
% LR - learning rate
% E - epoch size
%
% Outputs:
% W - weights of trained network
% b - bias of trained network
% J_plot - cost plot per epoch

    [W, b] = Param_init(h, N_h, size(X,2), length(unique(Y))); % initialization of parameters
    J_plot = zeros(1,E);

    for i = 1:E
        
        [~, J, Z, A] = ForwardPass(W, b, X, Y); % forward pass
        [dW, db] = backpropagation(W, X, Y, Z, A); % backward pass
        [W, b] = updateParams(W, b, dW, db, LR); % updating parameters
        J_plot(i) = J;
    end
end

