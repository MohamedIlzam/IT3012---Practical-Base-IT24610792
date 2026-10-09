# test_logic.py
"""
Test Suite for SE3062 Intelligent Systems Practical 07:
- KnowledgeBase Horn-clause logic engine and forward chaining
- SearchAgent A* search integration with logic deductions
- Safe and unsafe tile handling, fact isolation, and normal pathfinding preservation
- Multi-step simulation runs and visual grid environment validation
"""
import random
from logic_engine import KnowledgeBase
from agent import SearchAgent
from visual_grid_game import VisualGridHuntGame


def test_worksheet_logic_engine():
    """
    Part 4 Required Worksheet Test Cases:
    1. TargetVisible + HasDust must infer SafeToEngage but not Retreat.
    2. TargetVisible + HasDust + BloodseekerMissing must infer Retreat.
    Prints 'All Logic Engine Test Cases Passed!' on success.
    """
    kb = KnowledgeBase()

    # Register rules specified in Lab 07
    # Rule 1: ['TargetVisible', 'HasDust'] => 'SafeToEngage'
    # Rule 2: ['SafeToEngage', 'BloodseekerMissing'] => 'Retreat'
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    # Test 1: TargetVisible + HasDust
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.forward_chain()

    assert 'SafeToEngage' in kb.facts, "Test 1 Failed: SafeToEngage should be inferred."
    assert 'Retreat' not in kb.facts, "Test 1 Failed: Retreat should NOT be inferred without BloodseekerMissing."

    # Test 2: Add BloodseekerMissing -> Should now infer Retreat
    kb.tell_fact('BloodseekerMissing')
    kb.forward_chain()

    assert 'Retreat' in kb.facts, "Test 2 Failed: Retreat should be inferred when BloodseekerMissing is present."

    print("All Logic Engine Test Cases Passed!")


def test_rules_survive_clear_facts_and_no_leakage():
    """
    Validation: Confirm rules survive clear_facts() and facts do not leak between tiles.
    """
    kb = KnowledgeBase()
    kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
    kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    # Tile A facts
    kb.tell_fact('TargetVisible')
    kb.tell_fact('HasDust')
    kb.tell_fact('BloodseekerMissing')
    kb.forward_chain()
    assert 'Retreat' in kb.facts

    # Clear facts for next tile
    kb.clear_facts()
    assert len(kb.facts) == 0, "Facts were not cleared."
    assert len(kb.rules) == 2, "Rules did not survive clear_facts."

    # Tile B facts (benign tile)
    kb.tell_fact('RandomFact')
    kb.forward_chain()
    assert 'SafeToEngage' not in kb.facts, "Fact leakage from Tile A detected."
    assert 'Retreat' not in kb.facts, "Fact leakage from Tile A detected."


def test_astar_unsafe_tile_avoidance_and_safe_tile():
    """
    Validation:
    - Confirm an unsafe tile (1, 0) is never included in a valid path when alternatives exist.
    - Confirm a safe tile (0, 1) can be considered.
    """
    agent = SearchAgent()
    grid_size = (3, 3)
    start = (0, 0)
    goal = (1, 1)
    walls = []

    tile_facts = {
        (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],  # Dangerous -> deduces Retreat
        (0, 1): ['TargetVisible', 'HasDust']                        # Safe -> deduces SafeToEngage
    }

    path = agent.astar_search(start, goal, walls, grid_size, tile_facts=tile_facts)
    assert path is not None, "A* failed to find a valid path around the unsafe tile."

    # Confirm unsafe tile (1, 0) is never included in path
    assert (1, 0) not in path, "Unsafe tile (1, 0) was included in path coordinates."
    assert (1, 0) not in path.coordinates, "Unsafe tile (1, 0) present in path.coordinates."
    assert path[0] != 'Right', "First move was 'Right', which steps directly into unsafe tile (1, 0)."

    # Confirm safe tile (0, 1) was considered and traversed
    assert (0, 1) in path, "Safe tile (0, 1) was not traversed."
    assert (0, 1) in path.coordinates, "Safe tile (0, 1) was not in path.coordinates."
    assert path[0] == 'Up', "Expected initial step 'Up' onto safe tile (0, 1)."
    assert path == ['Up', 'Right'], f"Expected path ['Up', 'Right'], got {list(path)}."


def test_astar_no_tile_facts_preserves_normal_behavior():
    """
    Validation: Confirm no tile facts preserves normal A* behavior.
    """
    agent = SearchAgent()
    grid_size = (3, 3)
    start = (0, 0)
    goal = (1, 0)
    walls = []

    # With tile_facts=None, direct step to (1, 0) should be taken normally
    path_none = agent.astar_search(start, goal, walls, grid_size, tile_facts=None)
    assert path_none == ['Right'], f"Expected ['Right'], got {list(path_none)}."

    # With empty tile_facts dictionary
    path_empty = agent.astar_search(start, (1, 1), walls, grid_size, tile_facts={})
    assert path_empty is not None
    assert len(path_empty) == 2, f"Expected 2-step optimal path, got {len(path_empty)}."


def test_astar_unreachable_goal():
    """
    Validation: Confirm an unreachable goal is handled correctly (returns None or empty).
    """
    agent = SearchAgent()
    grid_size = (3, 3)
    start = (0, 0)
    goal = (2, 2)

    # 1. Goal blocked by walls
    walls = [(0, 1), (1, 0)]
    path_blocked = agent.astar_search(start, goal, walls, grid_size)
    assert path_blocked is None or len(path_blocked) == 0, "Blocked goal should return None or empty."

    # 2. All exit routes blocked by Retreat hazard tiles
    tile_facts = {
        (0, 1): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],
        (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing']
    }
    path_hazard_blocked = agent.astar_search(start, goal, walls=[], grid_size=grid_size, tile_facts=tile_facts)
    assert path_hazard_blocked is None or len(path_hazard_blocked) == 0, "Hazard-blocked goal should return None or empty."


def test_sense_and_act_game_integration():
    """
    Validation: Confirm SearchAgent.sense_and_act() integrates with A* and sample tile facts.
    """
    agent = SearchAgent()
    percept = {
        'agent_pos': (0, 0),
        'food_positions': {(1, 1)},
        'walls': [],
        'grid_size': (3, 3)
    }

    # By default, sense_and_act uses the sample tile facts from the worksheet:
    # (1, 0): Retreat, (0, 1): SafeToEngage
    action = agent.sense_and_act(percept)
    assert action == 'Up', f"sense_and_act should avoid unsafe (1, 0) and move 'Up', got {action}."


def test_multi_step_detour_validation():
    """
    Validation: Multi-step navigation test.
    The agent starts at (0, 0) and needs to reach food at (2, 0).
    Direct path along row 0 would go through (1, 0), but (1, 0) is unsafe (Retreat).
    The agent must take the safe detour through row 1: (0, 0) -> (0, 1) -> (1, 1) -> (2, 1) -> (2, 0).
    """
    agent = SearchAgent()
    env = VisualGridHuntGame(width=5, height=5, num_food=0, num_opponents=0, custom_walls=set())
    env.food_positions = {(2, 0)}
    env.agent_pos = [0, 0]

    visited = [tuple(env.agent_pos)]
    steps = 0
    max_steps = 15

    while not env.is_done() and (2, 0) in env.food_positions and steps < max_steps:
        percept = env.get_percept()
        action = agent.sense_and_act(percept)
        env.execute_action(action)
        current_pos = tuple(env.agent_pos)
        visited.append(current_pos)
        steps += 1

        # The agent must NEVER step onto (1, 0)
        assert current_pos != (1, 0), f"Agent violated safety and moved onto unsafe tile (1, 0) at step {steps}!"

    assert (1, 0) not in visited, "Unsafe tile (1, 0) was visited during detour."
    assert (0, 1) in visited, "Safe tile (0, 1) was not used in detour."
    assert (2, 0) not in env.food_positions, "Agent failed to collect target food."


def test_multi_step_simulated_runs():
    """
    Validation: Multiple full simulated game runs over 30+ steps each.
    Confirms the agent never steps onto (1, 0), handles dynamic food collection,
    and never crashes across diverse random seeds.
    """
    seeds = [42, 101, 777]
    for seed in seeds:
        random.seed(seed)
        agent = SearchAgent()
        env = VisualGridHuntGame(width=8, height=8, num_food=8, num_opponents=0, custom_walls=[(3, 3), (3, 4)])

        # Ensure food at (1, 0) is not spawned to keep the test strictly about navigation
        if (1, 0) in env.food_positions:
            env.food_positions.remove((1, 0))

        steps = 0
        max_steps = 35

        while not env.is_done() and steps < max_steps:
            percept = env.get_percept()
            action = agent.sense_and_act(percept)
            env.execute_action(action)
            steps += 1

            # Strict verification: Agent must never step onto (1, 0)
            assert tuple(env.agent_pos) != (1, 0), (
                f"Seed {seed}: Agent stepped onto unsafe tile (1, 0) at step {steps}!"
            )

        assert env.score >= 0, f"Seed {seed}: Unexpected negative score {env.score}."


if __name__ == '__main__':
    print("=== Running Lab 07 Logic Engine & A* Integration Tests ===")
    test_worksheet_logic_engine()
    test_rules_survive_clear_facts_and_no_leakage()
    test_astar_unsafe_tile_avoidance_and_safe_tile()
    test_astar_no_tile_facts_preserves_normal_behavior()
    test_astar_unreachable_goal()
    test_sense_and_act_game_integration()
    test_multi_step_detour_validation()
    test_multi_step_simulated_runs()
    print("=== All Integration Validation Tests Passed Successfully! ===")
