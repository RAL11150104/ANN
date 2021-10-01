function x= backprop_softmax(x)

    x = x.*(1-x);

end