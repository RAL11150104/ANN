function [dW,db] = backpropagation(weights,bias,input,output,Z,A)
    
    % dE = @(y,y_hat)(-1*y./y_hat); 
    % dA = @(x)(x.*(1-x));
    
    dW = weights;
    db = bias;
    weights_total = length(weights);
    back_error = (1/size(output,2))*(A{end} - output);
    for i = 1:weights_total
        if i == weights_total
            dW{end-(i-1)} = input * back_error';
        else
            dW{end-(i-1)} = A{end-i} * back_error';
        end
        
        db{end-(i-1)} = sum(back_error,2);
        if i == weights_total
            continue
        else
            back_error = (weights{end-(i-1)}*back_error).*BF_relu(Z{end-i}); 
        end
    end
    
%     dE = @(y,y_hat)(-1*y./y_hat); 
%     dA = @(x)(x.*(1-x));
%     
%     back_error_1 = dE(output,A{end}).*dA(A{end});
%     dW1 = A{end-1} * back_error_1';
%     db1 = ones(size(bias{end},1),1).*sum(back_error_1,2);
%     
%     back_error_2 = (weights{end}*back_error_1).*BF_relu(Z{end-1});
%     dW2 = A{end-2} * back_error_2';
%     db2 = ones(size(bias{end-1},1),1).*sum(back_error_2,2);
%     
%     back_error_3 = (weights{end-1}*back_error_2).*BF_relu(Z{end-2});
%     dW3 = input * back_error_3';
%     db3 = ones(size(bias{end-2},1),1).*sum(back_error_3,2);
%     
%     dW = {dW3,dW2,dW1};
%     db = {db3,db2,db1};
    

end