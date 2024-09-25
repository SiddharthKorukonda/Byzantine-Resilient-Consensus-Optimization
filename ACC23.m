classdef Agent
    properties
        value        % Current value of the agent
        neighbors    % Set of neighbors
        objective    % Local objective function
        byzantine    % Boolean to identify if the agent is Byzantine
    end
    
    methods
        function obj = Agent(initial_value, neighbors, objective, byzantine)
            obj.value = initial_value;
            obj.neighbors = neighbors;
            obj.objective = objective;
            obj.byzantine = byzantine;
        end
        
        function subgrad = subgradient(obj, x)
            % Returns the subgradient of the agent's local objective function at x
            subgrad = obj.objective.subgradient(x);
        end
        
        function filtered_values = filter_values(obj, received_values, f)
            % Filters out the f furthest values (Byzantine filtering)
            distances = vecnorm(obj.value - received_values, 2, 2);
            [~, sorted_indices] = sort(distances);
            filtered_values = received_values(sorted_indices(1:end-f), :);
        end
        
        function obj = update_value(obj, received_values, alpha, f)
            % Filter Byzantine agents and update the agent's value
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
                        % Gather values from neighbors
                        neighbor_values = vertcat(obj.agents(agent.neighbors).value);
                        % Update agent's value
                        agent = agent.update_value(neighbor_values, obj.alpha, obj.f);
                    end
                    new_values(i, :) = agent.value;
                end
                % Update all agents' values
                for i = 1:length(obj.agents)
                    obj.agents(i).value = new_values(i, :);
                end
            end
        end
    end
end

% Example usage:
num_agents = 5;
dimension = 2;
alpha = 0.01;
f = 1; % Number of Byzantine agents to filter out

agents = Agent.empty(num_agents, 0);
for i = 1:num_agents
    initial_value = rand(1, dimension); % Random initial values
    neighbors = setdiff(1:num_agents, i); % All other agents are neighbors
    objective = @(x) sum(x.^2); % Simple quadratic objective function for each agent
    byzantine = (i == 1); % First agent is Byzantine
    agents(i) = Agent(initial_value, neighbors, objective, byzantine);
end

consensus_system = ConsensusSystem(agents, alpha, f);
consensus_system.run_optimization(100);

for i = 1:num_agents
    fprintf('Agent %d final value: [%f, %f]\n', i, agents(i).value(1), agents(i).value(2));
end
