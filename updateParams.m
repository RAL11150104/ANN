function [W, b] = updateParams(W, b, dW, db, LR)
% this function updates the wights and biases using the computed gradients
% dW and db.

    W = W - LR*dW;
    b = b - LR*db;
end

