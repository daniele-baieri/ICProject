function [switch_conf, input_middlePerm] = MINselfRouting(middlePerm, flagPrint)
st = dbstack;
name = st(1).name;

if flagPrint
    fprintf('Starting script %s...\n\n', name);
    fprintf('%s - middlePerm in input\n', name);
    disp(middlePerm)
end

N_input = max(size(middlePerm));
N_nodes = N_input/2;
n_stagesHalf = log2(N_input);

switch_conf = zeros(N_nodes, n_stagesHalf);
route = zeros(N_input,2*n_stagesHalf);
route(:,1) = 0:N_input-1;

middlePosition = zeros(N_input);
for h = 1:N_input
    middlePosition(middlePerm(h)+1) = h-1;
end

for k = 1:n_stagesHalf
    for h = 1:2:N_input
        binPosition = de2bi(middlePosition(route(h,2*k-1)+1),n_stagesHalf,'left-msb');
        binNode = de2bi(ceil(h/2)-1,n_stagesHalf-1,'left-msb');
        
        if binPosition(k)
            switch_conf(ceil(h/2),k) = 1;
            route(h,2*k)   = route(h+1,2*k-1);
            route(h+1,2*k) = route(h,2*k-1);
        else
            route(h,2*k)   = route(h,2*k-1);
            route(h+1,2*k) = route(h+1,2*k-1);
        end
       
        if k < n_stagesHalf
            if binNode(k)
                route(h-2^(n_stagesHalf-k)+1,2*k+1) = route(h,2*k);
                route(h+1,2*k+1) = route(h+1,2*k);
            else
                route(h+2^(n_stagesHalf-k),2*k+1) = route(h+1,2*k);
                route(h,2*k+1) = route(h,2*k);
            end
        end
   end
end

if flagPrint
    fprintf('%s - Routing on the first MIN:\n', name);
    disp(route)
    fprintf('%s - Switch Configuration:\n', name);
    disp(switch_conf);
end

input_middlePerm = route(:,2*n_stagesHalf-1)';
