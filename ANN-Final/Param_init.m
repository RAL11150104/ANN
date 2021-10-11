% PARAMETER INITIALIZATION

% Sample Data

% h = 3;
% Nh = 14;
% Ni = 784;
% No = 3;
% load('Y_train.mat');
% load('X_train.mat');

function [W,b] = Param_init(h,Nh,Ni,No)
    % Initialization of Weights and Bias
    W = {1,h+1};
    b = {1,h};
    % Generation of random numbers (Weights), zero vectors (Bias), and 
    % scaling by He's Initialization (Fan-in only)
    for i = 1:h+1
        if i == 1
            He_init = sqrt(2/(Ni));
            W{i} = rand(Nh,Ni).*He_init; 
            b{i} = zeros(Nh,1);
        elseif i == h+1
            He_init = sqrt(2/(Nh));
            W{i} = rand(Nh,No).*He_init;
        else
            He_init = sqrt(2/(Nh));
            W{i} = rand(Nh,Nh).*He_init;
            b{i} = zeros(Nh,1);
        end
    end

end