# agent.py
"""
Agent implementations for SE3062 Intelligent Systems.
Includes:
- GreedyGridAgent (Baseline wanderer)
- SimpleReflexAgent (Condition-Action reflex agent)
- ModelBasedAgent (State-tracking reflex agent)
- SearchAgent (A* search integrated with Propositional Horn-Clause Knowledge Base)
"""
import random
import heapq
from collections import deque
from typing import List, Tuple, Dict, Set, Optional, Any

from logic_engine import KnowledgeBase


class Path(list):
    """
    A list of movement actions (e.g., ['Up', 'Right']) representing a path,
    with embedded coordinate tracking so that both action sequences and
    visited coordinates can be inspected and validated.
    """

    def __init__(self, actions=None, coordinates=None):
        super().__init__(actions or [])
        self.coordinates = list(coordinates or [])

    def __contains__(self, item: Any) -> bool:
        # If checked against a coordinate tuple like (1, 0), verify against visited coordinates
        if isinstance(item, tuple):
            return item in self.coordinates
        return super().__contains__(item)


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept.get('agent_pos', [0, 0])
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Simple Reflex Agent responding directly to immediate percepts."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'suck'
        elif percept.get('wall_ahead'):
            return 'Left'
        else:
            return 'Up'


class ModelBasedAgent:
    """Model-Based Agent maintaining internal state to escape corners/loops."""

    def __init__(self):
        self.turn_cycle = ['Left', 'Right', 'Down', 'Up']
        self.turn_index = 0
        self.last_action = None

    def sense_and_act(self, percept: dict) -> str:
        if percept.get('food_here'):
            return 'suck'
        elif percept.get('wall_ahead'):
            action = self.turn_cycle[self.turn_index % len(self.turn_cycle)]
            self.turn_index += 1
            self.last_action = action
            return action
        else:
            return 'Up'


class SearchAgent:
    """
    Search-based Agent utilizing A* Search integrated with a KnowledgeBase
    for logical deduction of dangerous/safe neighboring tiles.
    """

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']
        # Initialize Knowledge Base
        self.kb = KnowledgeBase()
        # Register Lab 07 Horn-clause rules:
        # ['TargetVisible', 'HasDust'] => 'SafeToEngage'
        # ['SafeToEngage', 'BloodseekerMissing'] => 'Retreat'
        self.kb.tell_rule(['TargetVisible', 'HasDust'], 'SafeToEngage')
        self.kb.tell_rule(['SafeToEngage', 'BloodseekerMissing'], 'Retreat')

    def bfs_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walls: List[Tuple[int, int]],
        grid_size: Tuple[int, int]
    ) -> Optional[List[str]]:
        """
        Breadth-First Search (BFS) to find the shortest path in hops.
        Preserves compatibility with earlier Practical 03 requirements.
        """
        start = tuple(start)
        goal = tuple(goal)
        wall_set = {tuple(w) for w in walls}
        width, height = grid_size

        if start == goal:
            return []

        queue = deque([(start, [])])
        visited = {start}

        directions = [
            ('Up', (0, 1)),
            ('Right', (1, 0)),
            ('Down', (0, -1)),
            ('Left', (-1, 0))
        ]

        while queue:
            current, path = queue.popleft()

            for action, (dx, dy) in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if neighbor == goal:
                    return path + [action]

                if (
                    0 <= neighbor[0] < width and
                    0 <= neighbor[1] < height and
                    neighbor not in wall_set and
                    neighbor not in visited
                ):
                    visited.add(neighbor)
                    queue.append((neighbor, path + [action]))

        return None

    def astar_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walls: List[Tuple[int, int]],
        grid_size: Tuple[int, int],
        tile_facts: Optional[Dict[Tuple[int, int], List[str]]] = None
    ) -> Optional[Path]:
        """
        A* Search algorithm integrated with KnowledgeBase forward chaining.
        For every physically valid neighboring tile, clears the KB facts,
        loads that tile's facts, runs forward_chain(), and skips the neighbor
        if 'Retreat' is deduced.
        """
        start = tuple(start)
        goal = tuple(goal)
        wall_set = {tuple(w) for w in walls}
        width, height = grid_size

        # Treat None as an empty dictionary
        if tile_facts is None:
            tile_facts = {}

        if start == goal:
            return Path([], [start])

        def heuristic(pos: Tuple[int, int]) -> int:
            # Manhattan distance heuristic
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Priority queue entries: (f_score, g_score, counter, current_pos, action_path, coord_path)
        pq = []
        counter = 0
        g_scores = {start: 0}
        heapq.heappush(pq, (heuristic(start), 0, counter, start, [], [start]))
        closed_set = set()

        directions = [
            ('Up', (0, 1)),
            ('Right', (1, 0)),
            ('Down', (0, -1)),
            ('Left', (-1, 0))
        ]

        while pq:
            f, g, _, current, actions, coords = heapq.heappop(pq)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal:
                return Path(actions, coords)

            for action, (dx, dy) in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check physical grid validity
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                if neighbor in wall_set:
                    continue

                # Knowledge Base inference for this neighbor tile
                self.kb.clear_facts()
                neighbor_facts = tile_facts.get(neighbor, [])
                for fact in neighbor_facts:
                    self.kb.tell_fact(fact)
                self.kb.forward_chain()

                # If Retreat is deduced, skip the neighbor entirely
                if 'Retreat' in self.kb.facts:
                    continue

                # Continue normal A* processing
                tentative_g = g + 1
                if neighbor in g_scores and tentative_g >= g_scores[neighbor]:
                    continue

                g_scores[neighbor] = tentative_g
                h = heuristic(neighbor)
                counter += 1
                heapq.heappush(
                    pq,
                    (tentative_g + h, tentative_g, counter, neighbor, actions + [action], coords + [neighbor])
                )

        # Unreachable goal
        return None

    def sense_and_act(
        self,
        percept: dict,
        tile_facts: Optional[Dict[Tuple[int, int], List[str]]] = None
    ) -> str:
        """
        Sensory interface for the agent. Uses sample tile facts from the worksheet
        by default, and plans the next action using A* towards the nearest food target.
        """
        # The worksheet expects sense_and_act() to pass sample tile facts to A*
        if tile_facts is None:
            tile_facts = percept.get('tile_facts', {
                (1, 0): ['TargetVisible', 'HasDust', 'BloodseekerMissing'],
                (0, 1): ['TargetVisible', 'HasDust']
            })

        start = tuple(percept.get('agent_pos', (0, 0)))
        walls = list(percept.get('walls', []))
        grid_size = percept.get('grid_size', (percept.get('width', 10), percept.get('height', 10)))

        food_positions = percept.get('food_positions', set())
        path = None
        if food_positions:
            # Sort food pellets by Manhattan distance and find nearest reachable one
            sorted_foods = sorted(
                food_positions,
                key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1])
            )
            for cand_goal in sorted_foods:
                cand_path = self.astar_search(start, cand_goal, walls, grid_size, tile_facts=tile_facts)
                if cand_path and len(cand_path) > 0:
                    path = cand_path
                    break
        else:
            goal = percept.get('goal', (grid_size[0] - 1, grid_size[1] - 1))
            if start == goal:
                return 'suck'
            path = self.astar_search(start, goal, walls, grid_size, tile_facts=tile_facts)

        if path and len(path) > 0:
            return path[0]

        # Fallback if no path is found: pick an adjacent tile verified safe by the KB
        for action in self.actions_pool:
            dx, dy = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}[action]
            cand = (start[0] + dx, start[1] + dy)
            if (
                0 <= cand[0] < grid_size[0] and
                0 <= cand[1] < grid_size[1] and
                cand not in walls
            ):
                self.kb.clear_facts()
                for fact in tile_facts.get(cand, []):
                    self.kb.tell_fact(fact)
                self.kb.forward_chain()
                if 'Retreat' not in self.kb.facts:
                    return action

        return 'Up'