clc;
clear all;

addpath("activation functions");

load("X_train.mat");
load("Y_train.mat");

load("X_test.mat");
load("Y_test.mat");

N_o = 3;
fnn = [120,84];
h = size(fnn,2);

f_size = [5,2,5,2];
skip_size = [1,2,1,2];
f_num = [6,6,16,16];
conv_seq = [1,0,1,0];
pad_size = [0,0,0,0];

E = 1000;
LR = 0.0001;

%% initializing parameters
[f_cnn, b_cnn, W_fnn, b_fnn] = initParameters(f_size, f_num, skip_size, pad_size, conv_seq, fnn, X_train, N_o);

J_plot_train = zeros(1,E);
% J_plot_test = zeros(1,E);

Acc_plot_train = zeros(1,E);
% Acc_plot_test = zeros(1,E);

j = 1;

figure();
hold on;
J_curve_train = plot(1:j,J_plot_train(1:j));
% J_curve_test = plot(1:j,J_plot_test(1:j));
Acc_curve_train = plot(1:j,Acc_plot_train(1:j));
% Acc_curve_test = plot(1:j,Acc_plot_test(1:j));
% legend('training error','test error','training accuracy','test accuracy','FontSize',15,'location','northwest');


% X_train = (X_train - mean(X_train))./std(X_train);


for j = 1:E
tic
%% converting the f_cnn weights into linear weights
[W_cnn] = f2W(X_train, f_cnn, skip_size, pad_size, conv_seq);

%% training
[Z, A, J_train, Y_out_train] = forwardPass(X_train, Y_train, f_cnn, b_cnn, f_size, skip_size, pad_size, conv_seq, W_fnn, b_fnn);

Y_hat = zeros(size(Y_train,1),length(unique(Y_train)));
for i = 1:size(Y_hat,1)
    Y_hat(i,Y_train(i)+1) = 1;
end
Y = Y_hat.*Y_out_train;
Y = sum(Y,2);
Y(Y<0.5) = 0;
Y(Y>= 0.5) = 1;
Acc_plot_train(j) = sum(Y,'all')*100/size(Y_out_train,1);

% %% testing
% [~, ~, J_test, Y_out_test] = forwardPass(X_test, Y_test, f_cnn, b_cnn, f_size, skip_size, pad_size, conv_seq, W_fnn, b_fnn);
% 
% Y_hat = zeros(size(Y_test,1),length(unique(Y_test)));
% for i = 1:size(Y_hat,1)
%     Y_hat(i,Y_test(i)+1) = 1;
% end
% 
% Y = Y_hat.*Y_out_test;
% Y = sum(Y,2);
% Y(Y<0.5) = 0;
% Y(Y>= 0.5) = 1;
% Acc_plot_test(j) = sum(Y,'all')*100/size(Y_out_test,1);

%% backward pass
[dW_lcnn, db_lcnn, dW_fnn, db_fnn] = backwardPass(X_train, Y_train, A, Z, W_cnn, b_cnn, W_fnn, b_fnn, pad_size, conv_seq);

%% converting linear weights to f_cnn weights
[df_cnn, db_cnn] = W2f(dW_lcnn, db_lcnn, f_size, skip_size, pad_size, conv_seq);

% updating parameters
[f_cnn, b_cnn, W_fnn, b_fnn] = updateParameters(LR, conv_seq, df_cnn, db_cnn, dW_fnn, db_fnn, f_cnn, b_cnn, W_fnn, b_fnn);
t = toc;

J_plot_train(j) = J_train;
% J_plot_test(j) = J_test;

delete(J_curve_train);
% delete(J_curve_test);
delete(Acc_curve_train);
% delete(Acc_curve_test);
hold on;
xlabel("Epoch",'FontSize',32);

yyaxis left
ylabel("Error",'fontsize',32);
J_curve_train = plot(1:j,J_plot_train(1:j),'r','linestyle','-','marker','none','linewidth',2);
% J_curve_test = plot(1:j,J_plot_test(1:j),'b','linestyle','-','marker','none','linewidth',2);

yyaxis right
ylabel("Accuracy (%)",'fontsize',32);
Acc_curve_train = plot(1:j,Acc_plot_train(1:j),'r','linestyle','--','marker','none','linewidth',2);
% Acc_curve_test = plot(1:j,Acc_plot_test(1:j),'b','linestyle','--','marker','none','linewidth',2);
% legend([J_curve_train,J_curve_test,Acc_curve_train,Acc_curve_test],'training error','test error','training accuracy','test accuracy','FontSize',15,'location','northwest');

pause(0.001);
clc;

disp("finished training for epoch: " + j)
disp("Cost of training is: " + J_train)
disp("Finished iterating in: " + t + " seconds")
end

% X_test = X_test;

% [Y_out, ~, ~, ~] = forward_pass(W, b, X_train, Y_train);
% Y_hat = zeros(size(Y_train,1),length(unique(Y_train)));
% for i = 1:size(Y_hat,1)
%     Y_hat(i,Y_train(i)+1) = 1;
% end
% 
% Y = Y_hat.*Y_out;
% Y = sum(Y,2);
% Y(Y<0.5) = 0;
% Y(Y>= 0.5) = 1;
% training_Accuracy = sum(Y,'all')*100/size(Y_out,1)
% 
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
