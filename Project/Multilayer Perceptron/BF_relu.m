function x = BF_relu(x)
    
    x(x>=0)=1;
    x(x<0)=0;
    
end