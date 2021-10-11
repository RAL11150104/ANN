function [dW, db] = gradCheck(W, b, X, Y)
% this function is used to check the weights and bias gradients derived via
% gradient decent.

% initializing dW and dB
for j = 1:size(W,1)
    dW{j} = zeros(size(W{j}));
    db{j} = zeros(size(b{j}));
end

epsilon = 10^-6;

for j = 1:size(W,2)
    for k = 1:size(W{j},1)
        for m = 1:size(W{j},2)
            W_plus = W;
            W_minus = W;
            W_plus{j}(k,m) = W_plus{j}(k,m) + epsilon;
            W_minus{j}(k,m) = W_minus{j}(k,m) - epsilon;
            dW{j}(k,m) = (fp(W_plus, b, X, Y) - fp(W_minus, b, X, Y))/(2*epsilon);
        end

            b_plus = b;
            b_minus = b;
            b_plus{j}(k,1) = b_plus{j}(k,1) + epsilon;
            b_minus{j}(k,1) = b_minus{j}(k,1) - epsilon;
            db{j}(k,1) = (fp(W, b_plus, X, Y) - fp(W, b_minus, X, Y))/(2*epsilon);
    end
end


%% Nested Functions
    
    function J = fp(W, b, X, Y)
        % This function simulates the forward pass in neural network
        % evaluation, where J is the cost/loss per instance of training
        % computed via the log loss function

        % initializing the binary class matrix counterpart of Y.
        Y_hat = zeros(size(Y,1),length(unique(Y)));
        for i = 1:size(Y_hat,1)
            Y_hat(i,Y(i)+1) = 1;
        end

        z = X*W{1}' + b{1}';
        a = relu(z);

        for i = 2:size(W,2)-1
            z = a*W{i}' + b{i}';
            a = relu(z);
        end
        
        z = a*W{i+1}' + b{i+1}';
        a = softmax(z')';
        
        Y_out = a;
        Y_out(Y_out == 0) = realmin; % approximating 0 as an infinitesimally small number; for numerical stability
        
        J = (-1/size(Y_hat,1))*sum(Y_hat.*log(Y_out),'all');
    end

    function a = relu(a)
    % This function simulates the RELU activation function.
    a(a<0) = 0;
    end

end

