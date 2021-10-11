function [dW,db] = backpropagation(W, X, Y, Z, A)
    
    for i = 1:length(W)
        W{i} = W{i}';
        Z{i} = Z{i}';
        A{i} = A{i}';
    end
    X = X';
    Y = Y';
    
    dW = cell(1,length(W));
    db = cell(1,length(W));
    
    weights_total = length(W);
    back_error = (1/size(Y,2))*(A{end} - Y);
    for i = 1:weights_total
        if i == weights_total
            dW{end-(i-1)} = X * back_error';
        else
            dW{end-(i-1)} = A{end-i} * back_error';
        end
        
        db{end-(i-1)} = sum(back_error,2);
        if i == weights_total
            continue
        else
            back_error = (W{end-(i-1)}*back_error).*d_relu(Z{end-i}); 
        end
    end
    
    for i = 1:length(dW)
        dW{i} = dW{i}';
    end
    
end