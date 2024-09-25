classdef Agent
    properties
        value        
        neighbors    
        objective    
        byzantine    
    end
    
    methods
        function obj = Agent(initial_value, neighbors, objective, byzantine)
            obj.value = initial_value;
            obj.neighbors = neighbors;
            obj.objective = objective;
            obj.byzantine = byzantine;
        end
        
        function subgrad = subgradient(obj, x)
            subgrad = obj.objective.subgradient(x);
        end
        
        function filtered_values = filter_values(obj, received_values, f)

            distances = vecnorm(obj.value - received_values, 2, 2);
            [~, sorted_indices] = sort(distances);
            filtered_values = received_values(sorted_indices(1:end-f), :);
        end
        
        function obj = update_value(obj, received_values, alpha, f)
            
            retained_values = obj.filter_values(received_values, f);
            consensus_value = mean(retained_values, 1);
            obj.value = consensus_value - alpha * obj.subgradient(consensus_value);
        end
    end
end

classdef ConsensusSystem
    properties
        agents
        alpha
        f
    end
    
    methods
        function obj = ConsensusSystem(agents, alpha, f)
            obj.agents = agents;
            obj.alpha = alpha;
            obj.f = f;
        end
        
        function run_optimization(obj, num_iterations)
            for t = 1:num_iterations
                new_values = zeros(length(obj.agents), length(obj.agents(1).value));
                for i = 1:length(obj.agents)
                    agent = obj.agents(i);
                    if ~agent.byzantine
                   
                        neighbor_values = vertcat(obj.agents(agent.neighbors).value);
                     
                        agent = agent.update_value(neighbor_values, obj.alpha, obj.f);
                    end
                    new_values(i, :) = agent.value;
                end
                
                for i = 1:length(obj.agents)
                    obj.agents(i).value = new_values(i, :);
                end
            end
        end
    end
end


num_agents = 5;
dimension = 2;
alpha = 0.01;
f = 1; 

agents = Agent.empty(num_agents, 0);
for i = 1:num_agents
    initial_value = rand(1, dimension); 
    neighbors = setdiff(1:num_agents, i); 
    objective = @(x) sum(x.^2); 
    byzantine = (i == 1); 
    agents(i) = Agent(initial_value, neighbors, objective, byzantine);
end

consensus_system = ConsensusSystem(agents, alpha, f);
consensus_system.run_optimization(100);

for i = 1:num_agents
    fprintf('Agent %d final value: [%f, %f]\n', i, agents(i).value(1), agents(i).value(2));
end
