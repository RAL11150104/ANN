function [prediction,error,Z,A] = feedforward(weights,bias,input,output)
    
    temp = input;
    weights_total = length(weights);
    for i = 1:weights_total
       Z{i} = weights{i}' * temp + bias{i};
       if i == weights_total
          A{i} = AF_softmax(Z{i});
          prediction = A{i} + 1e-15;
       else
          A{i} = AF_relu(Z{i});
       end
       temp = A{i};
    end
    
    constant = (1/size(output,2));
    error = (-1*constant)*sum(output.*log(prediction),'all');
    
end