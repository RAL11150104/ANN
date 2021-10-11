% EVALUATION

% INPUT: (FINAL) WEIGHT MATRIX, BIAS VECTOR, INPUTS
% OUTPUT: Yo

function [Yo] = Eval_ANN(W,b,X_test)
    %%Initialization
    [~,L] = size(W);
    Z = cell(1,L);
    A = cell(1,L);
    X_test = X_test./255; %Data Normalization
    %%Forward-pass
    for i = 1:L
        if i == 1
            Z{i} = X_test*W{1,i}.'+b{1,i}.';
            A{i} = relu(Z{i});  %%300x14
        elseif i == L
            Z{i} = A{i-1}*W{1,i};
            A{i} = softmax(Z{i}.'); % To make output into probabilities
            Yo = A{L}.'; %%300x3
        else
            Z{i} = A{i-1}*W{1,i}.'+b{1,i}.'; 
            A{i} = relu(Z{i}); %%300x14
        end
    end
end
