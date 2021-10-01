function [dW,dB] = backward_pass(weights,bias,input_layer,output_layer,Z,A,prediction)

    % Output to Second Hidden Layer Weights %

    back_error_1 = -1.*output_layer.*(prediction.^-1).*backprop_softmax(prediction);
    dW1 = back_error_1' * A{2};
    dB1 = ones(1,size(back_error_1,1)) * back_error_1;
    % Second Hidden Layer to First Hidden Layer Weights %

    back_error_2 = back_error_1 * weights{3};
    dW2 = (back_error_2 .* backprop_relu(A{2}))' * A{1};
    dB2 = ones(1,size(back_error_2,1)) * back_error_2;

    % First Hidden Layer to Input Weights %

    back_error_3 = back_error_2 * weights{2};
    dW3 = (back_error_3 .* backprop_relu(A{1}))' * input_layer;
    dB3 = ones(1,size(back_error_3,1)) * back_error_3;
    
    
    dW = {dW3,dW2,dW1};
    dB = {dB3,dB2,dB1};
    
end