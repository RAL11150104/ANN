function [W_cnn] = f2W(X, f_cnn, skip_size, pad_size, conv_seq)
    W_cnn = cell(1, size(f_cnn,2));

    N_i = size(X,2);
    N_z = size(X,3);
    for i = 1:size(f_cnn,2)
        f_size = size(f_cnn{i}{end},2);
        f_num = size(f_cnn{i},2);
        N_h = ((N_i - f_size + 2*pad_size(i))/skip_size(i)) + 1;

        for j = 1:f_num
            W_cnn{i}{j} = zeros(N_i, N_h, N_z);

            if(conv_seq(i) == 1) % convolution
                filter_transpose = zeros(f_size, 1, N_z);
                for n = 1:size(f_cnn{i}{j},3)
                    filter_transpose(:,:,n) = f_cnn{i}{j}(:,:,n)';
                end

                for n = 1:N_h
                    s = (n-1)*skip_size(i) + 1;
                    W_cnn{i}{j}(s:(s + f_size-1),n,:) = filter_transpose;
                end
            elseif(conv_seq(i) == 0) % pooling (maxpool)
                W_cnn{i}{j} = 1;
            end
        end
        N_z = f_num;
        N_i = N_h;

    end
end

