function [weights,bias,loss,prediction] = trainNN(input,output,LR,epoch,weights,bias)
   
    for i = 1:epoch
        [prediction,error,Z,A] = feedforward(weights,bias,input,output);
        [dW,db] = backpropagation(weights,bias,input,output,Z,A);
        [weights,bias] = parameter_update(weights,bias,dW,db,LR);
        loss(i) = error;
    end
    
end