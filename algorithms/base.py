import math
from typing import List, Set, Tuple

import numpy as np

from graph import Graph


class MaxCliqueSolver:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.best_solution = []
        self.maximum_clique_size = 0
        self.branch_num = 0
        self.eps = 1e-5
        self.branch_num = 0

    # https://stackoverflow.com/questions/59009712/
    def is_clique(self, nodes: List[int]) -> bool:
        subgraph = self.graph.graph.subgraph(nodes)
        num_nodes = len(nodes)
        return subgraph.size() == num_nodes * (num_nodes - 1) / 2

    def _add_constraint(self, branching_var_idx: int, right_hand_side: List[int], current_branch: int) -> None:
        self.cplex_model.linear_constraints.add(
            lin_expr=[[[f'x{branching_var_idx}'], [1.0]]],
            senses=['E'],
            rhs=right_hand_side,
            names=[f'c{current_branch}'],
        )

    def add_left_constraint(self, branching_var: Tuple[int, float], current_branch: int) -> None:
        branching_var_idx, branching_var_value = branching_var
        branching_var_value = 0 if branching_var_value < self.eps else branching_var_value
        right_hand_side = [math.floor(branching_var_value)]
        self._add_constraint(
            branching_var_idx=branching_var_idx,
            right_hand_side=right_hand_side,
            current_branch=current_branch
        )

    def add_right_constraint(self, branching_var: Tuple[int, float], current_branch: int) -> None:
        branching_var_idx, branching_var_value = branching_var
        right_hand_side = [math.ceil(branching_var_value)]
        self._add_constraint(
            branching_var_idx=branching_var_idx,
            right_hand_side=right_hand_side,
            current_branch=current_branch
        )

    def current_solution_is_best(self, current_objective_value):
        current_objective_value = (
            math.ceil(current_objective_value)
            if not math.isclose(
                current_objective_value,
                round(current_objective_value),
                rel_tol=self.eps,
            )
            else current_objective_value
        )
        return current_objective_value > self.maximum_clique_size

    def get_branching_var(self, current_values: List[float]) -> Tuple[int, float]:
        if all([math.isclose(x, np.round(x), rel_tol=self.eps) for x in current_values]):
            return -1

        not_integer_vars = [
            (idx, x)
            for idx, x in enumerate(current_values)
            if not math.isclose(x, np.round(x), rel_tol=self.eps)
        ]
        return max(not_integer_vars, key=lambda x: x[1])

    def init_model_with_heuristic_solution(self) -> None:
        best_heuristic_sol = self.initial_heuristic()
        if not self.is_clique(list(best_heuristic_sol)):
            raise Exception('Initial heuristic solution is not clique!')

        solution = np.zeros(len(self.graph.nodes))
        solution[list(best_heuristic_sol)] = 1
        self.best_solution = list(solution)
        self.maximum_clique_size = len(best_heuristic_sol)

    def initial_heuristic(self) -> Set[int]:
        best_clique = set()
        for vertex in self.graph.nodes:
            current_clique = set()
            current_clique.add(vertex)
            vertexes_degree = {
                vertex: self.graph.graph.degree(vertex)
                for vertex in self.graph.graph.neighbors(vertex)
            }
            while True:
                max_degree_vertex = max(
                    vertexes_degree,
                    key=vertexes_degree.get,
                )
                current_clique.add(max_degree_vertex)

                max_degree_vertex_neighbors = {
                    vertex: self.graph.graph.degree(vertex)
                    for vertex in self.graph.graph.neighbors(max_degree_vertex)
                }

                vertexes_degree = {
                    vertex: vertexes_degree[vertex]
                    for vertex in set(vertexes_degree).intersection(
                        set(max_degree_vertex_neighbors),
                    )
                }
                if not vertexes_degree:
                    break

            if len(current_clique) > len(best_clique):
                best_clique = current_clique
        return best_clique
