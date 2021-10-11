clc;
clear all;

load("X_train.mat");
load("Y_train.mat");

addpath("activation functions");
[W,b] = initParams(2, 14, size(X_train,2), length(unique(Y_train)));
% load("parameters")
% W = weights;
% b = bias;

E = 100;
LR = 0.0000001;

J_plot = zeros(1,E);
j = 1;

J_curve = plot(1:j,J_plot(1:j));

% X_train = (X_train - mean(X_train))./std(X_train);
% load("X_test.mat");
% load("Y_test.mat");

% X(Y(Y == 9),:) = [];
% Y(Y == 9) = [];

for j = 1:E
[Y_out, Z, A, J] = forward_pass(W, b, X_train, Y_train);
[dW, db] = backward_pass(W, X_train, Y_train, Z, A);
% [gc_dW, gc_db] = gradCheck(W, b, X_train, Y_train);
% 
% db{2}
% gc_db{2}
[W, b] = updateParams(W, b, dW, db, LR);

% X_test = X_test;

J_plot(j) = J;

delete(J_curve);
J_curve = plot(1:j,J_plot(1:j));
pause(0.001);

clc;
disp("finished training for epoch: " + j)
disp("Cost of training is: " + J)
end

% X_test = X_test;

[Y_out, ~, ~, ~] = forward_pass(W, b, X_train, Y_train);
Y_hat = zeros(size(Y_train,1),length(unique(Y_train)));
for i = 1:size(Y_hat,1)
    Y_hat(i,Y_train(i)+1) = 1;
end

Y = Y_hat.*Y_out;
Y = sum(Y,2);
Y(Y<0.5) = 0;
Y(Y>= 0.5) = 1;
training_Accuracy = sum(Y,'all')*100/size(Y_out,1)

% [Y_out, ~, ~, ~] = forward_pass(W, b, X_test, Y_test);
% Y_hat = zeros(size(Y_test,1),length(unique(Y_test)));
% for i = 1:size(Y_hat,1)
%     Y_hat(i,Y_test(i)+1) = 1;
% end
% 
% Y = Y_hat.*Y_out;
% Y = sum(Y,2);
% Y(Y<0.5) = 0;
% Y(Y>= 0.5) = 1;
% test_Accuracy = sum(Y,'all')*100/size(Y_out,1)
