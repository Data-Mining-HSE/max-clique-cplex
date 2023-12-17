import math
import time
from typing import List, Set, Tuple

import cplex
import networkx as nx
import numpy as np

from graph import Graph


class MaxCliqueSolver:
    def __init__(self, graph: Graph, debug_mode: bool = False) -> None:
        self.graph = graph
        self.best_solution = []
        self.maximum_clique_size = 0
        self.branch_num = 0
        self.eps = 1e-5
        self.branch_num = 0
        self.debug_mode = debug_mode
        self.start_time = 0

    def get_solution_nodes(self, values: List[float]) -> List[int]:
        return np.where(np.isclose(values, 1.0, atol=self.eps))[0].tolist()

    def setup_cplex_model(self) -> cplex.Cplex:
        nodes_amount = len(self.graph.nodes)
        obj = [1.0] * nodes_amount
        upper_bounds = [1.0] * nodes_amount
        lower_bounds = [0.0] * nodes_amount
        types = ['C'] * nodes_amount
        columns_names = [f'x{x}' for x in range(nodes_amount)]
    
        problem = cplex.Cplex()
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)
        problem.objective.set_sense(problem.objective.sense.maximize)

        problem.variables.add(
            obj=obj,
            lb=lower_bounds,
            ub=upper_bounds,
            types=types,
            names=columns_names,
        )
        return problem
    
    def independent_set_constraints(self) -> List[str]:
        """
        Returns list of constraints for all vertices in each independent set
        x_0 + x_1 + ... +  x_i  <= 1
        """
        constraints = []
        for ind_set in self.graph.independent_vertex_sets:
            constraint = [[f'x{i}' for i in ind_set], [1.0] * len(ind_set)]
            constraints.append(constraint)
        return constraints

    def not_connected_vertices_constraints(self) -> List[str]:
        """
        Returns list of constraints between not connected vertices
        x_i + x_j <=1
        """
        constraints = []
        for xi, xj in self.graph.not_connected_vertexes:
            contraint = [[f'x{xi}', f'x{xj}'], [1.0, 1.0]]
            constraints.append(contraint)
        return constraints

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
            names=[f'c{current_branch}']
        )

    def add_left_constraint(self, branching_var: Tuple[int, float], current_branch: int) -> None:
        branching_var_idx, branching_var_value = branching_var
        right_hand_side = [max(math.floor(branching_var_value), 0)]
        self._add_constraint(branching_var_idx, right_hand_side, current_branch)

    def add_right_constraint(self, branching_var: Tuple[int, float], current_branch: int) -> None:
        branching_var_idx, branching_var_value = branching_var
        right_hand_side = [math.ceil(branching_var_value)]
        self._add_constraint(branching_var_idx, right_hand_side, current_branch)

    def current_solution_is_best(self, current_objective_value: float) -> bool:
        current_objective_value = (
            math.ceil(current_objective_value) if not math.isclose(current_objective_value, round(current_objective_value),
                rel_tol=self.eps,
            )
            else current_objective_value
        )
        return current_objective_value > self.maximum_clique_size

    def get_branching_var(self, current_values: List[float]) -> Tuple[int, float]:
        if all([math.isclose(x, np.round(x), rel_tol=self.eps) for x in current_values]):
            return (-1, None)

        not_integer_vars = [
            (idx, x) for idx, x in enumerate(current_values)
            if not math.isclose(x, np.round(x), rel_tol=self.eps)
        ]
        return max(not_integer_vars, key=lambda x: x[1])

    def init_model_with_heuristic_solution(self) -> None:
        best_heuristic_sol = self.get_hueristic_clique()
        solution = np.zeros(len(self.graph.nodes))
        solution[list(best_heuristic_sol)] = 1
        self.best_solution = list(solution)
        self.maximum_clique_size = len(best_heuristic_sol)

    def get_hueristic_clique(self) -> Set[int]:
        best_clique = set()
        for clique in nx.find_cliques(self.graph.graph):
            if len(clique) > len(best_clique):
                best_clique = clique
                continue
            return best_clique

    def get_solution(self) -> Tuple[float, List[float]]:
        try:
            self.cplex_model.solve()
            # get the solution variables and objective value
            current_values = self.cplex_model.solution.get_values()
            current_objective_value = self.cplex_model.solution.get_objective_value()
            return current_objective_value, current_values
        except:
            return None, None
    
    def get_processing_time(self) -> int:
        return int(time.time() - self.start_time)

    def show_update_solution(self, new_solution: int) -> None:
        if self.debug_mode:
            print(f'[{self.get_processing_time()}] The best solution is updated as {new_solution}', flush=True)
