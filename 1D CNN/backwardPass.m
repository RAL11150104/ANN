function [dW_lcnn, db_lcnn, dW_fnn, db_fnn] = backwardPass(X, Y, A, Z, W_cnn, b_cnn, W_fnn, b_fnn, pad_size, conv_seq)
    % backward pass pseudo-weight matrix approach
    n_cnn = size(W_cnn,2); % number of cnn layers
    m = size(X,1);
    
    dW_lcnn = cell(1,size(W_cnn,2));
    db_lcnn = cell(1,size(b_cnn,2));
    
    dW_fnn = cell(1,size(W_fnn,2));
    db_fnn = cell(1,size(b_fnn,2));

    %% FNN backward pass
    Y_hat = zeros(size(Y,1),size(A{end},2));
    for i = 1:size(Y_hat,1)
        Y_hat(i,Y(i)+1) = 1;
    end

    dZ = (1/size(Y_hat,1))*(A{end} - Y_hat);
    dW_fnn{end} = dZ'*A{end-1};
    db_fnn{end} = sum(dZ,1)';

    dZ =  dZ*W_fnn{end}.*d_relu(Z{end-1});

    for j = (size(A,2) - 1):-1:(n_cnn + 1)
        
       dW_fnn{j - n_cnn} = dZ'*A{j-1};
       db_fnn{j - n_cnn} = sum(dZ,1)';
       if (j == (n_cnn + 1) && conv_seq(end) == 0)
            dZ =  dZ*W_fnn{j - n_cnn};
       else
            dZ =  dZ*W_fnn{j - n_cnn}.*d_relu(Z{j-1});
       end
    end
    
    %% CNN backward pass
    dZ = reshape(dZ,m,size(dZ,2)/size(W_cnn{end},2),size(W_cnn{end},2));
    
    if conv_seq(end) == 1
        db_lcnn{end} = sum(dZ,1);
        db_lcnn{end} = sum(db_lcnn{end},2);
    elseif conv_seq(end) == 0
        db_lcnn{end} = 0;
    end

    % reshaping last maxpool layer if needed
    if conv_seq(n_cnn) == 0
        Z{n_cnn} = reshape(Z{n_cnn},size(Z{n_cnn},1),size(Z{n_cnn},2)/size(W_cnn{n_cnn},2),size(W_cnn{n_cnn},2));
        A{n_cnn} = reshape(A{n_cnn},size(A{n_cnn},1),size(A{n_cnn},2)/size(W_cnn{n_cnn},2),size(W_cnn{n_cnn},2));
    end

    for i = n_cnn:-1:2
        % computation of weights

        if conv_seq(i) == 1

           for j = 1:size(dZ,3)
              for k = 1:size(A{i-1},3)
                  dW_lcnn{i}{j}(:,:,k) = (dZ(:,:,j)'*[zeros(m,pad_size(i)),A{i-1}(:,:,k),zeros(m,pad_size(i))])';
              end
           end
        elseif conv_seq(i) == 0
            dW_lcnn{i} = 0;
        end

       % computation of dZ
       if conv_seq(i) == 1
           dZ_p = zeros([size(Z{i-1},1),size(Z{i-1},2)+2*pad_size(i),size(Z{i-1},3),size(W_cnn{i},2)]);
           for j = 1:size(W_cnn{i}{end},3)
              for k = 1:size(W_cnn{i},2)
                 dZ_p(:,:,j,k) =  dZ(:,:,k)*W_cnn{i}{k}(:,:,j)';
              end
           end
           dZ_p = sum(dZ_p,4);

       elseif conv_seq(i) == 0
           dZ_p = zeros(size(A{i-1}));
           dWp = d_maxpool(A{i-1},A{i});

           for j = 1:size(dWp,2)
              for k = 1:size(dWp{end},3)

                  dZ_p(k,:,j) = dZ(k,:,j)*dWp{j}(:,:,k)';
              end
           end


       end

       dZ  = dZ_p;

       if conv_seq(i-1) == 1
          if i > 2
                dZ = dZ.*[zeros(m,pad_size(i),size(Z{i-1},3)),d_relu(Z{i-1}),zeros(m,pad_size(i),size(Z{i-1},3))];
          else
                dZ = dZ.*d_relu(Z{i-1});
          end
       end

       % computation of biases
       if conv_seq(i-1) == 1
           db_lcnn{i-1} = sum(dZ,1);
           db_lcnn{i-1} = sum(db_lcnn{i-1},2);
       elseif conv_seq(i-1) == 0
           db_lcnn{i-1} = 0;
       end
    end
    
    for j = 1:size(dZ,3)
      for k = 1:size(X,3)
          dW_lcnn{1}{j}(:,:,k) = (dZ(:,:,j)'*[zeros(m,pad_size(1)),X(:,:,k),zeros(m,pad_size(1))])';
      end
    end

end

