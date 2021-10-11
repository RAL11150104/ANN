function [a] = logsig(z)
% This function simulates the logistic sigmoid activation function.

    a = 1./(1 + exp(-z));
end
