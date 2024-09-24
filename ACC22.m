classdef Agent
    properties
        value
        convex_set
    end
    
    methods
        function obj = Agent(initial_value, convex_set)
            obj.value = initial_value;
            obj.convex_set = convex_set;
        end
        
        function retained_values = filter_values(obj, received_values, f)
            distances = vecnorm(obj.value - received_values, 2, 2);
            [~, sorted_indices] = sort(distances);
            retained_values = received_values(sorted_indices(1:end-f), :);
        end
        
        function projected_value = project_to_convex_set(obj, x)
            A = obj.convex_set.A;
            b = obj.convex_set.b;
            options = optimoptions('fmincon', 'Display', 'off');
            projected_value = fmincon(@(y) norm(y - x), x, A, b, [], [], [], [], [], options);
        end
        
        function obj = update_value(obj, retained_values, alpha)
            update_step = sum(retained_values - obj.value, 1);
            obj.value = obj.value + alpha * update_step;
            obj.value = obj.project_to_convex_set(obj.value);
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
        
        function new_values = run_consensus(obj, num_iterations)
            for iter = 1:num_iterations
                for i = 1:length(obj.agents)
                    agent = obj.agents(i);
                    received_values = vertcat(obj.agents([1:i-1, i+1:end]).value);
                    retained_values = agent.filter_values(received_values, obj.f);
                    obj.agents(i) = agent.update_value(retained_values, obj.alpha);
                end
            end
            
            new_values = vertcat(obj.agents.value);
        end
    end
end

% Example usage:
agents = Agent.empty(5, 0);
for i = 1:5
    initial_value = rand(1, 2); % 2D random initial values
    convex_set.A = eye(2); % A matrix for Ax <= b
    convex_set.b = ones(2, 1); % b vector for Ax <= b
    agents(i) = Agent(initial_value, convex_set);
end

consensus_system = ConsensusSystem(agents, 0.1, 1);
final_values = consensus_system.run_consensus(10);

for i = 1:length(agents)
    fprintf('Agent %d final value: [%f, %f]\n', i, agents(i).value(1), agents(i).value(2));
end
