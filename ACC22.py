import numpy as np
import networkx as nx
from scipy.optimize import minimize

class Agent:
    def __init__(self, initial_value, convex_set):
        self.value = initial_value 
        self.convex_set = convex_set 
    
    def filter_values(self, received_values, f):
        distances = [np.linalg.norm(self.value - rv) for rv in received_values]
        filtered_indices = np.argsort(distances)[:-f]
        return [received_values[i] for i in filtered_indices]
    
    def project_to_convex_set(self, x):
        A, b = self.convex_set
        cons = [{'type': 'ineq', 'fun': lambda x, A=A[i], b=b[i]: b - np.dot(A, x)} for i in range(len(b))]
        result = minimize(lambda x: np.linalg.norm(x - self.value), x, constraints=cons)
        return result.x
    
    def update_value(self, retained_values, alpha):
        update_step = np.sum([rv - self.value for rv in retained_values], axis=0)
        self.value += alpha * update_step
        self.value = self.project_to_convex_set(self.value)  

class ConsensusSystem:
    def __init__(self, agents, alpha, f):
        self.agents = agents
        self.alpha = alpha
        self.f = f
    
    def run_consensus(self, num_iterations):
        for _ in range(num_iterations):
            new_values = []
            for agent in self.agents:
                received_values = [a.value for a in self.agents if a != agent]
        
                retained_values = agent.filter_values(received_values, self.f)
               
                agent.update_value(retained_values, self.alpha)
            
            new_values.append([agent.value for agent in self.agents])
        
        return new_values

agents = [
    Agent(np.random.rand(2), (np.eye(2), np.ones(2))) for _ in range(5)
]  
consensus_system = ConsensusSystem(agents, alpha=0.1, f=1)

consensus_values = consensus_system.run_consensus(10)

for i, agent in enumerate(agents):
    print(f"Agent {i+1} final value: {agent.value}")
