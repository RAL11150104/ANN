% Forward Pass (ReLU)

% INPUT: Weight Matrix, Bias Vector, Input Vector, Input Labels
% Sample data
function [Yo, J, Z, A] = ForwardPass(W,b,X_train,Ydata)
    %% Initialization
    [~,L] = size(W);
    Z = cell(1,L);
    A = cell(1,L);
    X_train = X_train./255; % Normalization of input data
    
    %% Forward-pass
    for i = 1:L
        if i == 1 % First Layer
            Z{i} = X_train*W{1,i}.'+b{1,i}.';
            A{i} = relu(Z{i});  %% 300x14
        elseif i == L % Last Layer
            Z{i} = A{i-1}*W{1,i};
            A{i} = softmax(Z{i}.'); % To make output into probabilities
            Yo = A{L}.'; %% 300x3
        else %Middle Layer
            Z{i} = A{i-1}*W{1,i}.'+b{1,i}.'; 
            A{i} = relu(Z{i}); %% 300x14
        end
    end
    %% Error Calc
    % Assigning Y_hat:
    % Mapping Ydata (Expected outcome) to Y_hat to have the same dimensions
    % as Yo (Prediction)
    
    Y_hat = zeros(size(Yo)); 
    for i = 1:size(Ydata,1)
        if Ydata(i,1) == 0
            Y_hat(i,1) = 1;
        elseif Ydata(i,1) == 1
            Y_hat(i,2) = 1;
        elseif Ydata(i,1) == 2
            Y_hat(i,3) = 1;
        end
    end
    
    % Solving for Jtotal
    % Note that the mean takes the average of the whole column first
    % Therefore, it is already divided by the total number of samples
    % Adding the individual mean will yield the total mean error (logistic)
    J = sum(mean(-Y_hat.*log(Yo)));
end
