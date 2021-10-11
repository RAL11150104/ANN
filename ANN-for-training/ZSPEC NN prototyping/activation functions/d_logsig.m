function [a] = d_logsig(a)

a = logsig(a).*(1-logsig(a));

end

