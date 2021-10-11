%% MULTILAYER PERCEPTRON %%

clear all;
clc;
close all;
tic;

%% LOADING DATASET %%

load training_data.mat;
load test_data.mat;

%% NEURAL NETWORK ARCHITECTURE %%

% Using the formula of Yotov et al 2020 %
% the number of hidden nodes is %
% 14.95 round off to 14 %

hidden = [15,15];
input_train = (x_train./255)';
input_test = (x_test./255)';
output_train = zeros(10,length(y_train));
output_test = zeros(10,length(y_test));

for i = 1:size(output_train,2)
   output_train(y_train(i)+1,i) = 1; 
end

for i = 1:size(output_test,2)
   output_test(y_test(i)+1,i) = 1; 
end

clear x_test; clear x_train; 

%% HYPERPARAMETER INITIALIZATION %%

LR = [1e-1,1e-2,1e-3,1e-4,1e-5]; % Learning Rate: 1e-1,1e-2,1e-3,1e-4,1e-5 %
epoch = [500,1000,1500,2000,2500]; % Epoch: 500, 1000, 1500, 2000, 2500 % 
cost_train = {};
cost_test = {};
final_test_accuracy = [];
final_train_accuracy = [];

%% PARAMETER INITIALIZATION %%

for j = 1:length(epoch)
    for k = 1:length(LR)
        
[weights,bias] = parameter_initialization(input_train,hidden,output_train);

[weights,bias,loss_train,prediction_train] = trainNN(input_train,output_train,LR(k),epoch(j),weights,bias);

[prediction_test,loss_test,~,~] = feedforward(weights,bias,input_test,output_test);
[~,prediction_train] = max(prediction_train);
[~,prediction_test] = max(prediction_test);

prediction_train = prediction_train - 1;
prediction_test = prediction_test - 1;

train_accuracy = prediction_train - y_train';
test_accuracy = prediction_test - y_test';

counter_train = 0;
for i = 1:length(train_accuracy)
    if train_accuracy(i) == 0
       counter_train = counter_train + 1;
   end
end

counter_test = 0;
for i = 1:length(test_accuracy)
   if test_accuracy(i) == 0
       counter_test = counter_test + 1;
   end
end

train_accuracy = counter_train/length(train_accuracy);
test_accuracy = counter_test/length(test_accuracy);

%% FOR RECORDING %%

final_train_accuracy(k,j) = train_accuracy;
final_test_accuracy(k,j) = test_accuracy;
cost_train{k,j} = loss_train;
cost_test{k,j} = loss_test;

    end
end
toc

save('RESULTS_15_15','final_train_accuracy','final_test_accuracy','cost_train','cost_test');