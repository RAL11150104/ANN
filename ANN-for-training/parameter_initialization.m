function [weights,bias] =  parameter_initialization(inputs,hidden,output)
    
    nodes_input = size(inputs,1);
    nodes_hidden = hidden;
    nodes_output = size(output,1);
    
    nodes_total = [nodes_input,nodes_hidden,nodes_output];
    weights_total = length(nodes_total) - 1;
    
    for i = 1:weights_total
        weights{i} = normrnd(0,sqrt(2/nodes_total(i)),nodes_total(i),nodes_total(i+1));
        bias{i} = zeros(nodes_total(i+1),1);
    end
    
end