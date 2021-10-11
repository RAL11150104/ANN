function [dW, db] = backward_pass(W, X, Y, Z, A)
    j = size(W,2);
    
    Y_hat = zeros(size(Y,1),length(unique(Y)));
    for i = 1:size(Y_hat,1)
        Y_hat(i,Y(i)+1) = 1;
    end
    
    dZ{j} = (1/size(Y,1))*(A{end} - Y_hat);
    
    dW{j} = dZ{j}'*A{j-1};
    db{j} = sum(dZ{j},1)';

    dZ{j-1} =  dZ{j}*W{j}.*d_relu(Z{j-1});

    for j = (size(W,2) - 1):-1:2
       dW{j} = dZ{j}'*A{j-1};
       db{j} = sum(dZ{j},1)';
       dZ{j-1} =  dZ{j}*W{j}.*d_relu(Z{j-1});

    end

    dW{j-1} = (dZ{j-1}'*X);
    db{j-1} = sum(dZ{j-1},1)';

end

