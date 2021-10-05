function [a] = d_softmax(a)
    a = a.*(1-a);
end

