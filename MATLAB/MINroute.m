function route = MINroute(N_input,N_nodes,n_stagesHalf,n_stages,switch_conf)
route = zeros(N_input,2*n_stages);
route(:,1) = 0:N_input-1;

for j=1:n_stages-1                                         %1:n-1   j corrisponde agli stadi - se metto n-2 considero per N=8 solo i primi due stadi di switch
    for k=1:N_nodes                                     % for each node in a stage if node is set to cross, the exchange is realized
        if switch_conf(k,j)
            route(2*k,2*j) = route(2*k-1,2*j-1);
            route(2*k-1,2*j) = route(2*k,2*j-1);
        else
            route(2*k-1,2*j) = route(2*k-1,2*j-1);
            route(2*k,2*j) = route(2*k,2*j-1);
        end
    end
    for k=0:N_nodes-1                        % for each node in a stage the butterfly exchange related to the stage is realized computing the two new inputs
        bin_k = de2bi(k,n_stagesHalf-1,'left-msb');
        stage = mod(j-1,n_stagesHalf-1)+1;
        if bin_k(stage)
            route(2*k+1,2*j+1) = route(2*(k-2^(n_stagesHalf-1-stage))+2,2*j);
            route(2*k+2,2*j+1) = route(2*k+2,2*j);    
        else
            route(2*k+1,2*j+1) = route(2*k+1,2*j);
            route(2*k+2,2*j+1) = route(2*(k+2^(n_stagesHalf-1-stage))+1,2*j);     
        end
    end
end
for k=1:N_nodes                                         % for the last stage of nodes (that is stage n) if node is set to cross, the exchange is realized
    if switch_conf(k,n_stages)
        route(2*k,2*n_stages) = route(2*k-1,2*n_stages-1);
        route(2*k-1,2*n_stages) = route(2*k,2*n_stages-1);
    else
        route(2*k-1,2*n_stages) = route(2*k-1,2*n_stages-1);
        route(2*k,2*n_stages) = route(2*k,2*n_stages-1);
    end
end

% disp(route)
