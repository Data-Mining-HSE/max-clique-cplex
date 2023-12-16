import math
from typing import List, Tuple

import cplex
import numpy as np

from algorithms.base import MaxCliqueSolver
from graph import Graph


class BNBSolver(MaxCliqueSolver):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)
        self.cplex_model = self.construct_model()

    def construct_model(self) -> cplex.Cplex:
        problem = self.setup_cplex_model()

        independent_vertex_sets_amount = len(self.graph.independent_vertex_sets)
        not_connected_edges_amount = len(self.graph.not_connected_vertexes)

        right_hand_side = [1.0] * (not_connected_edges_amount + independent_vertex_sets_amount)
        constraint_names = [f'c{x}' for x in range(not_connected_edges_amount + independent_vertex_sets_amount)]
        constraint_senses = ['L'] * (not_connected_edges_amount + independent_vertex_sets_amount)

        constraints = [*self.independent_set_constraints(), *self.not_connected_vertices_constraints()]

        problem.linear_constraints.add(
            lin_expr=constraints,
            senses=constraint_senses,
            rhs=right_hand_side,
            names=constraint_names,
        )
        return problem

    def solve(self) -> None:
        self.init_model_with_heuristic_solution()

        self.branching()
        solution_nodes = np.where(np.isclose(self.best_solution, 1.0, atol=self.eps))
        self.is_solution_is_clique = self.is_clique(solution_nodes[0].tolist())

    def left_branching(self, branching_var: Tuple[int, float], cur_branch: int) -> None:
        self.add_left_constraint(branching_var, cur_branch)
        self.branching()
        self.cplex_model.linear_constraints.delete(f'c{cur_branch}')

    def right_branching(self, branching_var: Tuple[int, float], cur_branch: int) -> None:
        self.add_right_constraint(branching_var, cur_branch)
        self.branching()
        self.cplex_model.linear_constraints.delete(f'c{cur_branch}')

    def branching(self) -> None:
        current_objective_value, current_values = self.get_solution()
        if current_objective_value is None or not self.current_solution_is_best(current_objective_value):
            return

        if all(
            [
                math.isclose(x, np.round(x), rel_tol=self.eps)
                for x in current_values
            ],
        ):
            # Best Solution updated.
            self.best_solution = [round(x) for x in current_values]
            self.maximum_clique_size = math.floor(current_objective_value)
            return

        self.branch_num += 1
        cur_branch = self.branch_num
        branching_var = self.get_branching_var(current_values)

        if round(branching_var[1]): # closer to 1
            self.right_branching(branching_var, cur_branch)
            self.left_branching(branching_var, cur_branch)
        else: # closer to 0
            self.left_branching(branching_var, cur_branch)
            self.right_branching(branching_var, cur_branch)
