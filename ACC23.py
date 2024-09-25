import numpy as np

class Agent:
    def __init__(self, initial_value, neighbors, objective, byzantine=False):
        self.value = initial_value
        self.neighbors = neighbors
        self.objective = objective
        self.byzantine = byzantine
    
    def subgradient(self, x):
        # Returns the subgradient of the agent's objective function at x
        return self.objective.subgradient(x)
    
    def filter_values(self, received_values, f):
        # Filters out the f furthest values
        distances = [np.linalg.norm(self.value - rv) for rv in received_values]
        filtered_indices = np.argsort(distances)[:-f]
        return [received_values[i] for i in filtered_indices]
    
    def update_value(self, received_values, alpha, f):
        # Filter Byzantine agents and update the value using subgradient method
        retained_values = self.filter_values(received_values, f)
        consensus_value = np.mean(retained_values, axis=0)
        self.value = consensus_value - alpha * self.subgradient(consensus_value)

class ObjectiveFunction:
    def __init__(self, func, subgrad_func):
        self.func = func
        self.subgrad_func = subgrad_func
    
    def subgradient(self, x):
        return self.subgrad_func(x)

class ConsensusSystem:
    def __init__(self, agents, alpha, f):
        self.agents = agents
        self.alpha = alpha
        self.f = f
    
    def run_optimization(self, num_iterations):
        for _ in range(num_iterations):
            new_values = np.zeros((len(self.agents), len(self.agents[0].value)))
            for i, agent in enumerate(self.agents):
                if not agent.byzantine:
                    # Gather values from neighbors
                    neighbor_values = [self.agents[j].value for j in agent.neighbors]
                    # Update agent's value
                    agent.update_value(neighbor_values, self.alpha, self.f)
                new_values[i] = agent.value
            # Update all agents' values
            for i, agent in enumerate(self.agents):
                agent.value = new_values[i]

# Example usage:
num_agents = 5
dimension = 2
alpha = 0.01
f = 1 # Number of Byzantine agents to filter out

# Define the objective function for each agent
def objective_func(x):
    return np.sum(x**2)

def subgradient_func(x):
    return 2 * x

# Create agents
agents = []
for i in range(num_agents):
    initial_value = np.random.rand(dimension)
    neighbors = [j for j in range(num_agents) if j != i] # All other agents are neighbors
    objective = ObjectiveFunction(objective_func, subgradient_func)
    byzantine = (i == 0) # First agent is Byzantine
    agents.append(Agent(initial_value, neighbors, objective, byzantine))

# Create and run the consensus system
consensus_system = ConsensusSystem(agents, alpha, f)
consensus_system.run_optimization(100)

# Print final values
for i, agent in enumerate(agents):
    print(f"Agent {i+1} final value: {agent.value}")
