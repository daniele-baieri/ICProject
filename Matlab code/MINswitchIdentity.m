function switch_conf = MINswitchIdentity(N_input)

n_stagesHalf = log2(N_input);
n_stages = 2*n_stagesHalf - 1;
N_nodes = N_input/2;

switch_conf = zeros(N_nodes, n_stages);

for k = 2:n_stagesHalf-1
    items = 2^(n_stagesHalf-k-1);
    for h = items+1:2*items
        switch_conf(h,k) = 1;
    end
    
    copies = 2^(k-2);
    while copies >= 1
        start = N_nodes/(2*copies);
        for h = start+1:2*start
            switch_conf(h,k) = switch_conf(2*start-h+1,k);
        end
        copies = copies/2;
    end 
end
for h = 2:2:N_nodes/2
    switch_conf(h,n_stagesHalf) = 1;
    switch_conf(N_nodes+1-h,n_stagesHalf) = 1;
end
switch_conf(:,n_stagesHalf+1:n_stages) = switch_conf(:,2:n_stagesHalf);
% disp(switch_conf)
