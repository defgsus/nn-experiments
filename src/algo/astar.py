from typing import Callable, Optional, Any, List, Tuple, Union, Generator


def astar_search(
        start_node: Any,
        end_node: Any,
        adjacent_nodes_func: Union[
            Callable[[Any], List[Tuple[Any, float]]],
            Callable[[Any], Generator[Tuple[Any, float], None, None]],
        ],
        goal_cost_func: Optional[Callable[[Any, Any], float]] = None,
        verbose: bool = False,
):
    """
    Get optimal path between start_node and end_node.

    :param start_node: ID/handle of start node, e.g. (int, int)
    :param end_node: ID/handle of start node, e.g. (int, int)
    :param adjacent_nodes_func: function(Node) -> List[Tuple[Node, float]]
        function returning all accessible adjacent nodes in the world and their cost
        as list of (Node, float). Can also be a generator.
    :param goal_cost_func: function(Node, Node) -> float
        function returning heuristic cost between two positions

    :return: list of Nodes or None
    """
    if goal_cost_func is None:
        goal_cost_func = lambda p1, p2: 1.

    infinity = 2 << 31

    closed_set = set()
    open_set = {start_node}

    # cost of getting from start to this node
    g_score = {start_node: 0}

    # total cost if getting from start to end, through this node
    f_score = {end_node: goal_cost_func(start_node, end_node)}

    came_from = dict()

    while open_set:
        # pick smallest f from open_set
        current_node = None
        min_score = infinity
        for n in open_set:
            f = f_score.get(n, infinity)
            if f < min_score or current_node is None:
                min_score, current_node = f, n
        open_set.remove(current_node)

        if verbose:
            print(f"astar: at node {current_node}")

        # found!
        if current_node == end_node:
            path = [current_node]
            while current_node in came_from:
                current_node = came_from[current_node]
                path.append(current_node)
            return list(reversed(path))

        # flag as evaluated
        closed_set.add(current_node)

        for neighbor_node, step_cost in adjacent_nodes_func(current_node):

            if neighbor_node in closed_set:
                continue

            if neighbor_node not in open_set:
                open_set.add(neighbor_node)

            g = g_score.get(current_node) + step_cost
            # prune this path
            if g >= g_score.get(neighbor_node, infinity):
                continue

            # continue this path
            came_from[neighbor_node] = current_node
            g_score[neighbor_node] = g
            f_score[neighbor_node] = g + goal_cost_func(neighbor_node, end_node)

    return None

