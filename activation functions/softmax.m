function a = softmax( z )
% This function simulates the softmax activation function.

    max_a = max(z);
    log_b = z - (max_a + log(sum(exp(z-max_a))));
    a = exp(log_b);

end

