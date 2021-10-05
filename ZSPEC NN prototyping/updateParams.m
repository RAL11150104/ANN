function [W, b] = updateParams(W, b, dW, db, LR)
% this function updates the wights and biases using the computed gradients
% dW and db.
    for i = 1:size(W,2)
        W{i} = W{i} - LR*dW{i};
        b{i} = b{i} - LR*db{i};
    end
end

