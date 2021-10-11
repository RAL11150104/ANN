function [W, b] = updateParams(W, b, dW, db, LR)
% this function updates the wights and biases using the computed gradients
% dW and db.
    for j = 1:size(W,2)
        W{j} = W{j} - LR*dW{j};
        b{j} = b{j} - LR*db{j};
    end
end

