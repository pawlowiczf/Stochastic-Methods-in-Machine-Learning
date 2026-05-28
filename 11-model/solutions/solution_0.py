import numpy as np
from typing import Callable, List, Tuple

def new_metaheuristic(
    function: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    budget: int
) -> Tuple[float, np.ndarray]:
    """
    Dynamic Momentum Optimization (DMO) Algorithm.

    A novel metaheuristic that adapts individual agent coefficients (inertia,
    cognitive, social) based on their personal performance (improvement vs.
    stagnation) and includes a diversification strategy for highly stagnant agents.

    Args:
        function: The objective function to minimize. Takes a numpy.ndarray
                  (decision vector) and returns a float (objective value).
        bounds: A list of (lower, upper) pairs defining the search space for
                each dimension.
        budget: The total number of objective-function evaluations allowed.

    Returns:
        A tuple containing:
            - The best objective value found (float).
            - The corresponding decision vector (numpy.ndarray).
    """

    num_dimensions = len(bounds)
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])

    # --- DMO Parameters ---
    # Population size: heuristic based on dimensions, ensuring at least 2 agents
    # and considering budget limits.
    num_agents = max(5, min(50, 10 * num_dimensions))
    if budget < num_agents + 10: # Ensure enough budget for initial evaluations and some iterations
        num_agents = max(1, min(num_agents, budget // 2 if budget >= 2 else 1)) # If budget is 1, num_agents=1

    # Base coefficients for w, c1, c2
    w_base, c1_base, c2_base = 0.7, 1.5, 1.5

    # Ranges for adaptive coefficients
    w_min, w_max = 0.4, 0.9
    c1_min, c1_max = 0.5, 2.5
    c2_min, c2_max = 0.5, 2.5

    # Factors for adjusting coefficients on improvement/stagnation
    coef_growth_factor = 1.05
    coef_decay_factor = 0.95

    # Stagnation and diversification parameters
    stagnation_threshold = 10  # Iterations without personal best improvement before diversification
    perturbation_strength = 0.1 # Relative range for re-initializing stagnant agents around global_best

    # Maximum velocity component per dimension (to prevent jumps)
    # Scales with 20% of the dimension's search space width.
    v_max = (upper_bounds - lower_bounds) * 0.2

    # --- Initialization ---
    
    # Agents' current positions and velocities
    positions = np.random.uniform(lower_bounds, upper_bounds, (num_agents, num_dimensions))
    velocities = np.zeros((num_agents, num_dimensions))

    # Personal best for each agent
    personal_best_positions = np.copy(positions)
    personal_best_values = np.full(num_agents, np.inf)

    # Global best found so far
    global_best_value = np.inf
    global_best_position = np.zeros(num_dimensions)

    # Agent-specific adaptive coefficients, initialized to base values
    w_agents = np.full(num_agents, w_base)
    c1_agents = np.full(num_agents, c1_base)
    c2_agents = np.full(num_agents, c2_base)

    # Stagnation counter for each agent
    stagnation_counters = np.zeros(num_agents, dtype=int)

    evaluations_count = 0

    # Initial evaluation of all agents to set personal and global bests
    for i in range(num_agents):
        if evaluations_count >= budget:
            break
        
        current_val = function(positions[i])
        evaluations_count += 1

        personal_best_values[i] = current_val
        if current_val < global_best_value:
            global_best_value = current_val
            global_best_position = positions[i]
    
    # Handle cases where initial evaluations exhaust the budget
    if evaluations_count >= budget:
        return global_best_value, global_best_position

    # --- Main Optimization Loop ---
    # Continue until the budget of function evaluations is exhausted
    while evaluations_count < budget:
        for i in range(num_agents):
            if evaluations_count >= budget: # Check budget after each agent's turn
                break

            # Generate random numbers for cognitive and social components
            r1 = np.random.rand(num_dimensions)
            r2 = np.random.rand(num_dimensions)

            # Calculate cognitive and social components of velocity
            cognitive_component = c1_agents[i] * r1 * (personal_best_positions[i] - positions[i])
            social_component = c2_agents[i] * r2 * (global_best_position - positions[i])
            
            # Update velocity
            velocities[i] = w_agents[i] * velocities[i] + cognitive_component + social_component
            
            # Clamp velocity components to stay within defined limits
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # Update position
            positions[i] = positions[i] + velocities[i]
            
            # Clamp position to stay within the search bounds
            positions[i] = np.clip(positions[i], lower_bounds, upper_bounds)

            # Evaluate the new position
            current_val = function(positions[i])
            evaluations_count += 1

            # Update personal best and adapt coefficients
            if current_val < personal_best_values[i]:
                # Agent improved: update personal best and adapt coefficients for exploitation
                personal_best_values[i] = current_val
                personal_best_positions[i] = positions[i]
                
                c1_agents[i] = min(c1_max, c1_agents[i] * coef_growth_factor) # More self-confident
                w_agents[i] = max(w_min, w_agents[i] * coef_decay_factor)     # Less inertia (focus on new path)
                c2_agents[i] = max(c2_min, c2_agents[i] * coef_decay_factor)   # Less reliance on global
                
                stagnation_counters[i] = 0 # Reset stagnation counter
            else:
                # Agent did not improve: adapt coefficients for exploration
                c1_agents[i] = max(c1_min, c1_agents[i] * coef_decay_factor) # Less self-confident
                w_agents[i] = min(w_max, w_agents[i] * coef_growth_factor)   # More inertia (explore more widely)
                c2_agents[i] = min(c2_max, c2_agents[i] * coef_growth_factor) # More reliance on global (pull towards best known)
                
                stagnation_counters[i] += 1 # Increment stagnation counter
            
            # Update global best if a better solution is found
            if current_val < global_best_value:
                global_best_value = current_val
                global_best_position = positions[i]

            # Diversification for highly stagnant agents
            if stagnation_counters[i] > stagnation_threshold:
                # Perturb the agent's position around the global best
                perturb_range = (upper_bounds - lower_bounds) * perturbation_strength
                positions[i] = global_best_position + np.random.uniform(-perturb_range, perturb_range, num_dimensions)
                positions[i] = np.clip(positions[i], lower_bounds, upper_bounds)

                velocities[i] = np.zeros(num_dimensions) # Reset velocity
                stagnation_counters[i] = 0 # Reset stagnation counter
                
                # Reset adaptive coefficients to base values to give it a fresh start
                w_agents[i] = w_base
                c1_agents[i] = c1_base
                c2_agents[i] = c2_base
                
                # The reinitialized position will be evaluated in the agent's next turn.

    return global_best_value, global_best_position