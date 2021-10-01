function [prediction,loss,Z,A] = forward_pass(weights,bias,input_layer,output_layer)
    
    temp = input_layer;
    for i = 1:length(weights)
    
        Z{i} = temp * weights{i}' + bias{i};
        if i ~= length(weights)
            temp = relu(Z{i}); 
        else
            temp = softmax(Z{i});
        end
        A{i} = temp;
    
    end

    prediction = A{end} + 1e-15;
    error = (1/size(input_layer,1)) * (-1 * output_layer .* log(prediction));
    loss = sum(sum(error));

end