function [weights,bias] = parameter_update(weights,bias,dW,db,LR)
    
    weights_total = length(weights);
    for i = 1:weights_total
       weights{i} = weights{i}-LR.*dW{i};
       bias{i} = bias{i}-LR.*db{i};
    end
    
end