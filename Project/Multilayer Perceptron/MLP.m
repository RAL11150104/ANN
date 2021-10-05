%% MULTILAYER PERCEPTRON %%

clear all;
clc;
close all;

%% LOADING DATASET %%

load X_train.mat;
load Y_train.mat;

%% NEURAL NETWORK ARCHITECTURE %%

% Using the formula of Yotov et al 2020 %
% the number of hidden nodes is %
% 14.95 round off to 14 %

hidden = [14,14];
input = (X_train./255)';
output = [ones(100,1),zeros(100,1),zeros(100,1);zeros(100,1),ones(100,1),zeros(100,1);zeros(100,1),zeros(100,1),ones(100,1)]';

%% HYPERPARAMETER INITIALIZATION %%

LR = 0.00001; % Learning Rate %
loss = [];

%% PARAMETER INITIALIZATION %%

[weights,bias] = parameter_initialization(input,hidden,output);

%% FEEDFORWARD ALGORITHM %%
loss = [];

for i = 1:200
    
[prediction,error,Z,A] = feedforward(weights,bias,input,output);

%% BACKPROPAGATION ALGORITHM %%

[dW,db] = backpropagation(weights,bias,input,output,Z,A);
    
%% UPDATING PARAMETER %%

[weights,bias] = parameter_update(weights,bias,dW,db,LR);

loss = [loss,error];

end

