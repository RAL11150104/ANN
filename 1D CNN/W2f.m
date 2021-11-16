function [df_cnn, db_cnn] = W2f(dW_lcnn, db_lcnn, f_size, skip_size, pad_size, conv_seq)
    n_cnn = size(dW_lcnn,2);
    df_cnn = cell(1,n_cnn);
    db_cnn = cell(1,n_cnn);

    for i = 1:n_cnn
        if conv_seq(i) == 1
            for j = 1:size(dW_lcnn{i},2)

                if i < n_cnn
                    if(pad_size(i) ~= 0 && i ~= 1)
                        dW_lcnn{i}{j}(:,[1:pad_size(i+1),(size(dW_lcnn{i}{j},2)-pad_size(i+1)+1):size(dW_lcnn{i}{j},2)],:) = [];
                    end
                end
                
                dW_summer = zeros(f_size(i),size(dW_lcnn{i}{j},2),size(dW_lcnn{i}{j},3));
                for k = 1:size(dW_lcnn{i}{j},2)
                   s = (k-1)*skip_size(i) + 1; 

                   dW_summer(:,k,:) = dW_lcnn{i}{j}(s:(s + f_size(i) -1),k,:); 
                end
                
                % averaging criteria
                db_cnn{i}{j} = db_lcnn{i}(:,:,j);
                dW_summer2 = sum(dW_summer,2);
                
                for k = 1:size(dW_summer2,3)
                    df_cnn{i}{j} = dW_summer2(:,:,k)';
                end
                
            end

        elseif conv_seq(i) == 0
            df_cnn{i} = 0;
            db_cnn{i} = 0;

        end

    end
end

