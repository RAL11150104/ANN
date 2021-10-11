%Forward Pass (ReLU)

%INPUT: Weight Matrix, Bias Vector, Input Vector, Input Labels
%Sample data
function [Yo, J, Z, A] = ForwardPass(W, b, X, Y)
        % This function simulates the forward pass in neural network
        % evaluation, where J is the cost/loss per instance of training
        % computed via the log loss function

        % initializing the binary class matrix counterpart of Y.
        Y_hat = zeros(size(Y,1),length(unique(Y)));
        for i = 1:size(Y_hat,1)
            Y_hat(i,Y(i)+1) = 1;
        end
        
        Z{1} = X*W{1}' + b{1}';
        A{1} = relu(Z{1});

        for i = 2:size(W,2)-1
            Z{i} = A{i-1}*W{i}' + b{i}';
            A{i} = relu(Z{i});
        end
        
        Z{i+1} = A{i}*W{i+1}' + b{i+1}';
        A{i+1} = softmax(Z{i+1}')';
        
        Yo = A{i+1};
        Yo(Yo == 0) = realmin; % approximating 0 as an infinitesimally small number; for numerical stability
        
        
        J = (-1/size(Y_hat,1))*sum(Y_hat.*log(Yo),'all');
end
