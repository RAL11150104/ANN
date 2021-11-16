function [f_cnn, b_cnn, W_fnn, b_fnn] = initParameters(f_size, f_num, skip_size, pad_size, conv_seq, fnn, X, N_o)
    f_cnn = cell(1, size(f_size,2));
    b_cnn = cell(1, size(f_size,2));
    h = size(fnn,2);
    
    N_i = size(X,2);
    N_z = size(X,3);
    
    for i = 1:size(f_size,2)
        N_h = ((N_i - f_size(i) + 2*pad_size(i))/skip_size(i)) + 1;
        for j = 1:f_num(i)
            b_cnn{i}{j} = 0;
            if(conv_seq(i) == 1) % convolution
                f_cnn{i}{j} = normrnd(0,sqrt(2/(f_size(i)*N_z)),1, f_size(i), N_z );
            elseif(conv_seq(i) == 0) % pooling (maxpool)
                f_cnn{i}{j} = ones(1, f_size(i), N_z);
            end
        end
        N_z = f_num(i);
        N_i = N_h;
    end

    N_h = N_h*f_num(end);
    W_fnn = cell(1,size(fnn,2));
    b_fnn = cell(1,size(fnn,2));
    
    for i = 1:h
        if i == 1 % for cnn interface layer
            W_fnn{i} = normrnd(0,sqrt(2/N_h),fnn(i),N_h);
            b_fnn{i} = zeros(fnn(i),1);
        else % for layers in between
            W_fnn{i} = normrnd(0,sqrt(2/fnn(i-1)),fnn(i),fnn(i-1));
            b_fnn{i} = zeros(fnn(i),1);
        end
    end

    % for end layer
    W_fnn{h + 1} = normrnd(0,sqrt(2/fnn(i)),N_o,fnn(i));
    b_fnn{h + 1} = zeros(N_o,1);
    
end

