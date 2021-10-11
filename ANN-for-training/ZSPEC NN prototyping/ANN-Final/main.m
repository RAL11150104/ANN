clc;
clear all;

load("X_train.mat");
load("Y_train.mat");

addpath('activation functions');

X_train = X_train/255;

h = 2;
N_h = 14;
LR = 1e-2;
E = 100;

[W, b, J_plot] = trainNN(X_train, Y_train, h, N_h, LR, E);
J_curve = plot(J_plot);