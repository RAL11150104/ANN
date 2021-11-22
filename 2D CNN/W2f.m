function [df_cnn, db_cnn] = W2f(dW_lcnn, db_lcnn, f_size, skip_size, pad_size, conv_seq)
    n_cnn = size(dW_lcnn,2);
    df_cnn = cell(1,n_cnn);
    db_cnn = cell(1,n_cnn);

    for i = 1:n_cnn
        if conv_seq(i) == 1
            for j = 1:size(dW_lcnn{i},2)

                if i < n_cnn
                    if(pad_size(i) ~= 0 && i ~= 1)
                       
                       dW_lcnn{i}{j} = permute(dW_lcnn{i}{j},[2,3,1]);
                       dW_lcnn{i}{j} = reshape(dW_lcnn{i}{j}, sqrt(size(dW_lcnn{i}{j},1)), sqrt(size(dW_lcnn{i}{j},1)), size(dW_lcnn{i}{j},2), size(dW_lcnn{i}{j},3));
                       dW_lcnn{i}{j} =  dW_lcnn{i}{j}((pad_size(i+1)+1):(size(dW_lcnn{i}{j},2)-pad_size(i+1)),(pad_size(i+1)+1):(size(dW_lcnn{i}{j},2)-pad_size(i+1)),:,:);         
                       dW_lcnn{i}{j} = reshape(permute(dW_lcnn{i}{j},[1,2,4,3]),size(dW_lcnn{i}{j},2)^2,size(dW_lcnn{i}{j},4),size(dW_lcnn{i}{j},3));
                       dW_lcnn{i}{j} = permute(dW_lcnn{i}{j},[2,1,3]);
                        
                    end
                end
                
                dW_summer = zeros(f_size(i),f_size(i),size(dW_lcnn{i}{j},3),size(dW_lcnn{i}{j},2));
                N_h = sqrt(size(dW_lcnn{i}{j},2));
                for n = 1:N_h
                   s_n = (n-1)*skip_size(i) + 1;
                   for m = 1:N_h
                       s_m = (m-1)*skip_size(i) + 1; 
                       dW_i = dW_lcnn{i}{j}(:,(n-1)*N_h + m,:);
                       dW_i = reshape(dW_i,sqrt(size(dW_i,1)),sqrt(size(dW_i,1)),size(dW_i,3));
                       dW_summer(:,:,:,(n-1)*N_h + m) = dW_i(s_m:(s_m + f_size(i) -1),s_n:(s_n + f_size(i) -1),:); 
                   end
                end
                % averaging criteria
                db_cnn{i}{j} = db_lcnn{i}(:,:,j);
                dW_summer2 = sum(dW_summer,4);
                
                df_cnn{i}{j} = dW_summer2;
            end

        elseif conv_seq(i) == 0
            df_cnn{i} = 0;
            db_cnn{i} = 0;

        end

    end
end

