function [Z, A, J, Y_out] = forwardPass(X, Y, f_cnn, b_cnn, f_size, skip_size, pad_size, conv_seq, W_fnn, b_fnn)
    %% forward pass using Covn definition
    A_i = X;
    m = size(X,1);
    h = size(W_fnn,2)-1;
    Z = cell(1,size(f_size ,2)+ (h + 1));
    A = cell(1,size(f_size ,2)+ (h + 1));
    
    for i = 1:size(f_cnn,2)
       A_i = [zeros(m,pad_size(i),size(A_i,3)), A_i, zeros(m,pad_size(i),size(A_i,3))];
       N_h = ((size(A_i,2) - f_size(i))/skip_size(i)) + 1;
       
       Z_o = zeros(m,N_h,size(f_cnn{i},2));
       
       if(conv_seq(i) == 1) % convolution
           for j = 1:size(f_cnn{i},2)
               for k = 1:N_h
                   s = (k-1)*skip_size(i) + 1;
                   Z_o(:,k,j) = sum(sum(f_cnn{i}{j}.*A_i(:,s:(s + f_size(i) - 1),:) ,2),3) + b_cnn{i}{j};
               end
           end
           Z{i} = Z_o;
           A_i = relu(Z_o);
           A{i} = A_i;
       
       elseif(conv_seq(i) == 0) % pooling (max pooling)
           for j = 1:size(f_cnn{i},2)
               for k = 1:N_h
                   s = (k-1)*skip_size(i) + 1;
                   I_window = A_i(:,s:(s + f_size(i) - 1),j);
                   Z_o(:,k,j) = max(I_window,[],2);
               end
           end
           Z{i} = Z_o;
           A{i} = Z{i};
           A_i = A{i};
       end
       
    end
    
    Z{i} = reshape(Z{i},size(Z{i},1),size(Z{i},2)*size(Z{i},3),1); % unrolled feature space
    A{i} = reshape(A{i},size(A{i},1),size(A{i},2)*size(A{i},3),1); % unrolled feature space
    A_inp = A{i};
    n_cnn = size(f_size,2);
    
    for i = 1:h
       Z{i + n_cnn} = A_inp*W_fnn{i}' + b_fnn{i}';
       A{i + n_cnn} = relu(Z{i + n_cnn});
       A_inp = A{i + n_cnn};
    end
    
    Z{(i + 1) + n_cnn} = A_inp*W_fnn{i+1}' + b_fnn{i+1}';
    A{(i + 1) + n_cnn} = softmax(Z{(i + 1) + n_cnn}')';

    Y_out = A{(i + 1) + n_cnn};
    Y_out(Y_out == 0) = realmin; % approximating 0 as an infinitesimally small number; for numerical stability

    Y_hat = zeros(size(Y,1),size(A{end},2));
    for i = 1:size(Y_hat,1)
        Y_hat(i,Y(i)+1) = 1;
    end
    
    J = (-1/size(Y_hat,1))*sum(Y_hat.*log(Y_out),'all');
end
