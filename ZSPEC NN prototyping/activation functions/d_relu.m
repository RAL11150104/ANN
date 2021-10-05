function [a] = d_relu(a)
    
    a(a<0) = 0;
    a(a>=0) = 1;

end

