function a = AF_softmax(z)

    sum_f = sum(exp(z));
    a = exp(z)./sum_f;

end