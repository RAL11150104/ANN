function [W_cnn] = f2W(X, f_cnn, skip_size, pad_size, conv_seq)
    W_cnn = cell(1, size(f_cnn,2));

    N_i = sqrt(size(X,2));
    N_z = size(X,3);
    for i = 1:size(f_cnn,2)
        
        f_size = size(f_cnn{i}{end},2);
        f_num = size(f_cnn{i},2);
        N_i = N_i + 2*pad_size(i);
        N_h = ((N_i - f_size)/skip_size(i)) + 1;
        
        for j = 1:f_num
            if(conv_seq(i) == 1) % convolution
                W_cnn{i}{j} = zeros(N_i^2, N_h^2, N_z);
                filter = f_cnn{i}{j};
                
                for n = 1:N_h % column extrusion
                    s_n = (n-1)*skip_size(i) + 1;
                    for m = 1:N_h % row extrusion
                        s_m = (m-1)*skip_size(i) + 1;
                        W_i = zeros(N_i,N_i,N_z);
                        W_i(s_m:(s_m + f_size-1),s_n:(s_n + f_size-1),:) = filter;
                        W_i = reshape(W_i,N_i^2,1,N_z);
                        
                        W_cnn{i}{j}(:,(n-1)*N_h + m,:) = W_i;
                    end
                end
                
            elseif(conv_seq(i) == 0) % pooling (maxpool)
                W_cnn{i}{j} = 1;
            end
        end
        N_z = f_num;
        N_i = N_h;

    end
end

