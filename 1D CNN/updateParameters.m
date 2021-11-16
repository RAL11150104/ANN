function [f_cnn, b_cnn, W_fnn, b_fnn] = updateParameters(LR, conv_seq, df_cnn, db_cnn, dW_fnn, db_fnn, f_cnn, b_cnn, W_fnn, b_fnn)
    
    % updating CNN parameters
    for i = 1:size(df_cnn,2)
        if conv_seq(i) == 1
            for j = 1:size(df_cnn{i},2)
                f_cnn{i}{j} = f_cnn{i}{j} - LR*df_cnn{i}{j};
                b_cnn{i}{j} = b_cnn{i}{j} - LR*db_cnn{i}{j};
            end
        end
    end
    
    % updating FNN parameters
    for i = 1:size(dW_fnn,2)
       W_fnn{i} = W_fnn{i} - LR*dW_fnn{i}; 
       b_fnn{i} = b_fnn{i} - LR*db_fnn{i}; 
    end
end

