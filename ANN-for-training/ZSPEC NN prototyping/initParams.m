function [W,b] = initParams(h, N_h, N_i, N_o)

W = cell(1,h+1);
b = cell(1,h+1);

W{1} = normrnd(0,sqrt(2/N_i),N_h,N_i);
b{1} = zeros(N_h,1);

for i = 2:h
    W{i} = normrnd(0, sqrt(2/N_h), N_h, N_h);
    b{i} = zeros(N_h,1);
end

W{h+1} = normrnd(0, sqrt(2/N_h), N_o, N_h);
b{h+1} = zeros(N_o,1);

end

