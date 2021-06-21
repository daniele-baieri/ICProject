function switch_conf = MINswitchModifier(switch_conf, inizio, flagPrint)
st = dbstack;
name = st(1).name;
fprintf('Starting script %s...\n\n', name);

[N_nodes,n_stages] = size(switch_conf);
N_input = 2*N_nodes;
n_stagesHalf = (n_stages+1)/2;

if mod(inizio,2) == 0
    route = MINroute(N_input,N_nodes,n_stagesHalf,n_stages,switch_conf);
    middlePerm = mod(route(:,n_stages+1)'+inizio,N_input);
    switch_conf(:,1:n_stagesHalf) = MINselfRouting(middlePerm, flagPrint);
else
    switch_conf(:,n_stagesHalf+1) = 1 - switch_conf(:,n_stagesHalf+1);
    route = MINroute(N_input,N_nodes,n_stagesHalf,n_stages,switch_conf);
    middlePerm = route(:,n_stages+1)';
    newPerm = zeros(1,N_input);
    for h = 1:N_input
        item = route(h,2*n_stages);
        for k = 1:N_input
            if middlePerm(k) == item
                newPerm(k) = mod(inizio + h - 1, N_input);
                break
            end
        end
    end
    switch_conf(:,1:n_stagesHalf) = MINselfRouting(newPerm, flagPrint);
end