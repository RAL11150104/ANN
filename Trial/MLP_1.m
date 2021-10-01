%% MULTILAYER PERCEPTRON %%

% This is my personal Multilayer Perceptron %

%% LOADING DATASET %%

clear all; close all; clc;
load X_train.mat;
load Y_train.mat;

%% PARAMETER INITIALIZATION %%

% Using the formula of Yotov et al 2020; the number of hidden nodes is %
% 14.95 round off to 14 %

hidden_layer = [14,14];
input_layer = X_train./255;
output_layer = [ones(100,1),zeros(100,1),zeros(100,1);zeros(100,1),ones(100,1),zeros(100,1);zeros(100,1),zeros(100,1),ones(100,1)];
[weights,bias] = parameter_init(input_layer,hidden_layer,output_layer);

%% FORWARD PASS %%
ERROR = [];

for j = 1:200

[prediction,loss,Z,A] = forward_pass(weights,bias,input_layer,output_layer);

%% BACKWARD PASS %%

[dW,dB] = backward_pass(weights,bias,input_layer,output_layer,Z,A,prediction);

%% PARAMETER UPDATE %%

LR = 1e-7;
[weights,bias] = updateParams(weights,bias,dW,dB,LR);

ERROR = [ERROR;loss];

end

plot(ERROR)
