function [Z, A, J, Y_out] = forwardPass(X, Y, f_cnn, b_cnn, f_size, skip_size, pad_size, conv_seq, W_fnn, b_fnn)
    %% forward pass using Covn definition
    X = reshape(permute(X,[2,1,3]),sqrt(size(X,2)),sqrt(size(X,2)),size(X,3),size(X,1));
    A_i = X;
    
    m = size(X,4);
    h = size(W_fnn,2)-1;
    Z = cell(1,size(f_size ,2)+ (h + 1));
    A = cell(1,size(f_size ,2)+ (h + 1));
    
    for i = 1:size(f_cnn,2)
       A_i =  padarray(A_i,[pad_size(i),pad_size(i)],0,'both');
       N_h = ((size(A_i,2) - f_size(i))/skip_size(i)) + 1;
       Z_o = zeros(N_h,N_h,size(f_cnn{i},2),m);

       if(conv_seq(i) == 1) % convolution
           for j = 1:size(f_cnn{i},2) % per filter
               for k = 1:N_h % per column filter
                   s_k = (k-1)*skip_size(i) + 1;
                   for n = 1:N_h % per row filter
                       s_n = (n-1)*skip_size(i) + 1;     
                       
                       Z_o(n,k,j,:) = sum(sum(sum(f_cnn{i}{j}.*A_i(s_n:(s_n + f_size(i) - 1),s_k:(s_k + f_size(i) - 1),:,:) ,3),2),1) + b_cnn{i}{j};
                   end
               end
           end
            
           Z{i} = Z_o;
           A_i = relu(Z_o);
           A{i} = A_i;
       
       elseif(conv_seq(i) == 0) % pooling (max pooling)

           for j = 1:size(f_cnn{i},2)
               for k = 1:N_h
                   s_k = (k-1)*skip_size(i) + 1;
                   for n = 1:N_h
                       s_n = (k-1)*skip_size(i) + 1;
                       I_window = A_i(s_n:(s_n + f_size(i) - 1), s_k:(s_k + f_size(i) - 1),j,:);
                       Z_o(n,k,j,:) = max(I_window,[],[1,2]);
                   end
               end
           end
           Z{i} = Z_o;
           A{i} = Z{i};
           A_i = A{i};
       end

       Z{i} = reshape(permute(Z{i},[1,2,4,3]),size(Z{i},1)*size(Z{i},2),size(Z{i},4),size(Z{i},3));
       A{i} = reshape(permute(A{i},[1,2,4,3]),size(A{i},1)*size(A{i},2),size(A{i},4),size(A{i},3));
       Z{i} = permute(Z{i},[2,1,3]);
       A{i} = permute(A{i},[2,1,3]); 

       
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
