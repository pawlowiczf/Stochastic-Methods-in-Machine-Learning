The "Social Influence Optimization" (SIO) algorithm is a novel metaheuristic inspired by the dynamics of social networks and information diffusion. In a social network, individuals (agents) are influenced by their peers and prominent figures, leading to shifts in their opinions or behaviors. In SIO, agents (candidate solutions) navigate the search space by being attracted to more "influential" peers and the globally best solution found so far, combined with a degree of random exploration.

Here's how SIO works:

1.  **Initialization**: A population of `N` agents is randomly placed within the search bounds. Each agent's position represents a candidate solution. Their fitness (objective value) is evaluated. The best solution found among these initial agents is designated as the `global_best`.

2.  **Influence Calculation**: In each iteration (generation), the "influence" of each agent is calculated. Agents with lower fitness (better solutions for minimization) are considered more influential. This influence is normalized to create probabilities for peer selection.

3.  **Movement Rule**: For each agent, a new position is proposed based on three components:
    *   **Attraction to an Influential Peer**: Each agent `i` probabilistically selects another agent `j` from the population as its "influential peer." The probability of selecting an agent `j` is directly proportional to `j`'s influence. Agent `i` is then attracted towards `j`. This component promotes exploration around promising regions and leverages information from diverse good solutions.
    *   **Attraction to the Global Best**: Agent `i` is also attracted towards the `global_best` solution found across all generations. This component drives convergence towards the best-known optimum.
    *   **Random Perturbation**: A random vector is added to introduce exploration, helping agents escape local optima and explore new areas of the search space. The scale of this perturbation is typically related to the search space dimensions.

    These three components are weighted by `alpha`, `beta`, and `gamma` coefficients, respectively, to balance exploration and exploitation.

4.  **Boundary Handling**: After calculating the new position, it is clipped to ensure it remains within the defined search bounds.

5.  **Selection**: The proposed new position for agent `i` is evaluated. If its fitness is better than the agent's current fitness, the agent updates its position and fitness. Otherwise, it retains its current state. This is a form of `(1+1)` selection for each individual.

6.  **Global Best Update**: After all agents have potentially updated their positions, the `global_best` solution is updated if a new superior solution has been found.

7.  **Budget Management**: The algorithm continues these steps until the `budget` (total number of function evaluations) is exhausted. The `budget` is carefully managed to ensure the algorithm stops precisely when the allocated evaluations are used.

**Novelty Aspects**:
*   **Dynamic, Probabilistic Peer Influence**: Unlike algorithms that might select a fixed set of neighbors or use only the personal best/global best, SIO dynamically calculates influence based on the current population's fitness and uses this to probabilistically select a peer. This introduces a unique blend of local and global information sharing.
*   **Hybrid Movement Strategy**: The combination of attraction to a stochastically chosen influential peer, the global best, and scaled random perturbation provides a balanced mechanism for both broad exploration and directed exploitation.

```python
import numpy as np
from typing import Callable, List, Tuple

def social_influence_optimization(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int
) -> Tuple[float, np.ndarray]:
    """
    Social Influence Optimization (SIO) algorithm for black-box function minimization.

    SIO is a novel metaheuristic inspired by the spread of influence in a social network.
    Agents in the search space have 'influence' based on their fitness. They move
    by being attracted to more influential 'peers' and the global best solution,
    with an added random perturbation to promote exploration.

    Args:
        function: The objective function to minimize. It takes a NumPy array (decision vector)
                  and returns a float (objective value).
        bounds: A list of (lower, upper) pairs, defining the search space for each dimension.
        budget: The total number of objective function evaluations the algorithm may perform.

    Returns:
        A tuple containing:
            - The best objective value found (float).
            - The corresponding decision vector (NumPy array).
    """

    D = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    search_space_range = upper_bounds - lower_bounds

    # Handle the edge case of zero budget upfront
    if budget == 0:
        # If no evaluations are allowed, return infinity and a default position
        # (e.g., center of the search space if D > 0, or empty array if D = 0).
        if D > 0:
            return np.inf, (lower_bounds + upper_bounds) / 2
        else: # D=0 implies no search space, no decision vector.
            return np.inf, np.array([])

    # --- SIO Algorithm Parameters ---
    # Population size (N). It's capped at 50 to prevent excessive computational cost
    # per generation for very large budgets, and ensures N does not exceed the budget.
    # Minimum N is 1 to handle small budgets gracefully.
    N = min(budget, 50)
    N = max(1, N) # Ensure N is at least 1.

    # Influence and movement coefficients
    alpha = 0.5  # Attraction strength to the probabilistically selected influential peer
    beta = 0.3   # Attraction strength to the global best solution
    gamma = 0.1  # Scale factor for random perturbation
    epsilon = 1e-9 # Small constant to prevent division by zero in influence calculation

    # Initialize population randomly within bounds
    population = np.random.rand(N, D) * search_space_range + lower_bounds
    fitness = np.zeros(N)

    # Initialize global best
    best_global_val = np.inf
    best_global_pos = None

    eval_count = 0

    # Evaluate initial population
    for i in range(N):
        if eval_count >= budget: # Stop if budget runs out during initial evaluations
            break

        fitness[i] = function(population[i])
        eval_count += 1

        if fitness[i] < best_global_val:
            best_global_val = fitness[i]
            best_global_pos = population[i].copy()

    # If no evaluations were possible or global best wasn't set (e.g., budget=0, N=0, but already handled)
    # This case should not be reached with the current budget and N handling for budget > 0.
    if best_global_pos is None and eval_count > 0:
        # Fallback if somehow best_global_pos is not set, but evaluations happened
        best_global_val = np.min(fitness[:eval_count])
        best_global_pos = population[np.argmin(fitness[:eval_count])].copy()
    elif best_global_pos is None: # This should only happen if eval_count == 0, which implies budget == 0
        if D > 0: # This means budget was 0, and N was 0, so D was not 0
            return np.inf, (lower_bounds + upper_bounds) / 2
        else: # D=0 and budget=0
            return np.inf, np.array([])


    # Main optimization loop: continue until budget is exhausted
    while eval_count < budget:
        current_min_f = np.min(fitness)
        current_max_f = np.max(fitness)

        # Calculate influence for each agent
        # Influence is inversely proportional to fitness (lower fitness = higher influence)
        # Normalized to sum to 1 for probabilistic selection
        if current_max_f == current_min_f:
            # If all fitness values are identical, assign uniform influence
            influences = np.ones(N) / N
        else:
            influences = (current_max_f - fitness) / (current_max_f - current_min_f + epsilon)
            # Ensure influences sum to 1 for np.random.choice
            influences = influences / np.sum(influences)

        # Iterate through each agent to update its position
        for i in range(N):
            if eval_count >= budget:
                break # Stop if budget is exhausted mid-generation

            # --- 1. Select an influential peer for agent 'i' ---
            peer_pos = population[i] # Default: no peer influence (agent attracted to itself)

            if N > 1: # Peer selection only makes sense if there's more than one agent
                # Create a list of other agents' indices (excluding agent 'i' itself)
                peer_indices = np.delete(np.arange(N), i)
                # Get their corresponding influences
                peer_influences_for_selection = np.delete(influences, i)

                # Handle cases where all other peers have zero influence (e.g., due to identical fitness)
                if np.sum(peer_influences_for_selection) == 0:
                    # Fallback: select a peer uniformly at random
                    influential_peer_idx_original = np.random.choice(peer_indices)
                else:
                    # Normalize peer influences to sum to 1 for probabilistic selection
                    peer_influences_for_selection = peer_influences_for_selection / np.sum(peer_influences_for_selection)
                    influential_peer_idx_original = np.random.choice(peer_indices, p=peer_influences_for_selection)

                peer_pos = population[influential_peer_idx_original]

            # --- 2. Determine movement step ---
            # Directional vectors
            direction_to_peer = peer_pos - population[i] # If N=1, this is 0
            direction_to_global_best = best_global_pos - population[i]

            # Random perturbation. Scaled by a fraction of the total search space range
            # to make it robust across different problem scales.
            random_vec = np.random.randn(D) * search_space_range * gamma

            # Calculate the full step vector
            step = (alpha * direction_to_peer) + \
                   (beta * direction_to_global_best) + \
                   random_vec # Gamma is already applied to random_vec

            # --- 3. Update agent's position ---
            new_pos_i = population[i] + step

            # Apply boundary constraints: clip the position to stay within bounds
            new_pos_i = np.clip(new_pos_i, lower_bounds, upper_bounds)

            # --- 4. Evaluate the new position ---
            new_fitness_i = function(new_pos_i)
            eval_count += 1

            # If budget runs out immediately after this evaluation, handle it here
            if eval_count >= budget:
                if new_fitness_i < best_global_val:
                    best_global_val = new_fitness_i
                    best_global_pos = new_pos_i.copy()
                return best_global_val, best_global_pos

            # --- 5. Selection: Update agent if new position is better ---
            if new_fitness_i < fitness[i]:
                population[i] = new_pos_i.copy()
                fitness[i] = new_fitness_i
            # If not better, the agent keeps its current position and fitness

            # --- 6. Update global best solution found so far ---
            if fitness[i] < best_global_val:
                best_global_val = fitness[i]
                best_global_pos = population[i].copy()

    return best_global_val, best_global_pos
```
