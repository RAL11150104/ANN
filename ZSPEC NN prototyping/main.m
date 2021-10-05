clc;
clear all;

load("X_train.mat");
load("Y_train.mat");

addpath("activation functions")
[W,b] = initParams(2, 3, size(X_train,2), length(unique(Y_train)));

for j = 1:1
[Y_out, Z, A, J] = forward_pass(W, b, X_train, Y_train);
[dW, db] = backward_pass(W, X_train, Y_train, Z, A, Y_out);
[gc_dW, gc_db] = gradCheck(W, b, X_train, Y_train)
% [W, b] = updateParams(W, b, dW, db, 0.3);
% 
dW{3}
gc_dW{3}


J_plot(j) = J;
end

plot(J_plot)
