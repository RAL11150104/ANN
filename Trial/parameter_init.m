function [weights,bias] = parameter_init(input_layer,hidden_layer,output_layer)
    
    % Convention: Rows = Number of Samples; Columns = Number of Input Nodes %
    input_nodes = size(input_layer,2);
    output_nodes = size(output_layer,2);
    total_nodes = [input_nodes,hidden_layer,output_nodes];
    rng(1)
    
    for i = 1:length(hidden_layer)+1
        weights{i} = normrnd(0,sqrt(2/total_nodes(i)),total_nodes(i+1),total_nodes(i));
        bias{i} = zeros(total_nodes(i+1),1)';
    end
    
end