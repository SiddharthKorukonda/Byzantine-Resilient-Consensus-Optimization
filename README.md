# Byzantine-Resilient-Consensus-Optimization

# ACC22 - Resilient Constrained Consensus over Complete Graphs via Feasibility Redundancy
This paper addresses the problem of achieving a resilient consensus in a distributed multi-agent system under Byzantine faults, where some agents may behave maliciously or unpredictably. The focus is on scenarios where agents must reach a consensus while adhering to constraints defined by convex sets. The paper introduces a distributed algorithm that filters out potentially faulty agents and ensures the remaining normal agents reach a consensus on a value within their convex sets. Key contributions include conditions for ensuring feasibility and achieving exponentially fast consensus.


# ACC23 - Resilient Distributed Optimization
This paper studies distributed optimization in multi-agent systems where some agents (Byzantine agents) introduce faulty or malicious information. The goal is for the normal agents to cooperatively minimize a global objective function despite the presence of these faulty agents. The paper proposes a subgradient-based optimization algorithm that ensures resilient performance through graph and objective function redundancy. The algorithm guarantees that all normal agents converge to the same optimal point while mitigating the influence of Byzantine agents. It focuses on both theoretical guarantees and practical considerations for implementing resilient optimization in multi-agent systems.


# Summary
Both papers deal with resilience in multi-agent systems in the presence of Byzantine faults, but with different objectives. ACC22 focuses on constrained consensus—reaching an agreement under constraints, while ACC23 focuses on distributed optimization—minimizing a global objective function collaboratively. Both introduce algorithms that filter out faulty agents and provide theoretical guarantees for resilience.