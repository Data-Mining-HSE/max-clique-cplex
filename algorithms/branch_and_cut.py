import math
import time
from typing import List, Optional, Tuple

import cplex
import networkx as nx

from algorithms.base import MaxCliqueSolver
from graph import Graph


class BNCSolver(MaxCliqueSolver):
    def __init__(self, graph: Graph, tailing_off_time_threshold: int = 3600, debug_mode: bool =False):
        super().__init__(graph=graph, debug_mode=debug_mode)
        self.cplex_model = self.construct_model()
        self.tailing_off_time_threshold = tailing_off_time_threshold

    def add_multiple_constraints(self, constraints: List[int]) -> None:
        constraint_senses = ['L'] * (len(constraints))
        right_hand_side = [1.0] * (len(constraints))
        constraint_names = [f'c{x}' for x in range(len(constraints))]
        new_constraints = []

        for constraint in constraints:
            constraint = [
                [f'x{i}' for i in constraint],
                [1.0] * len(constraint),
            ]
            new_constraints.append(constraint)

        self.cplex_model.linear_constraints.add(
            lin_expr=new_constraints,
            senses=constraint_senses,
            rhs=right_hand_side,
            names=constraint_names,
        )

    def construct_model(self) -> cplex.Cplex:
        problem = self.setup_cplex_model()

        independent_vertex_sets_amount = len(self.graph.independent_vertex_sets)

        right_hand_side = [1.0] * independent_vertex_sets_amount
        constraint_names = [f'c{x}' for x in range(independent_vertex_sets_amount)]
        constraint_senses = ['L'] * independent_vertex_sets_amount

        constraints = self.independent_set_constraints()

        problem.linear_constraints.add(
            lin_expr=constraints,
            senses=constraint_senses,
            rhs=right_hand_side,
            names=constraint_names,
        )
        return problem

    def separation(self, solution, top_k: int = 2):
        independent_sets = self.graph.independent_sets_generation(
            minimum_set_size=2,
            iteration_number=1,
            max_weighted=True,
            solution=solution
        )
        sorted_set = sorted(
            independent_sets,
            key=lambda item: item[1],
            reverse=True,
        )
        top_k = min(top_k, len(sorted_set))
        new_constraints = [ind_set[0] for ind_set in sorted_set][:top_k]
        return new_constraints if len(new_constraints) else None

    def solve(self) -> None:
        self.init_model_with_heuristic_solution()
        self.start_time = time.time()
        self.branch_and_cut()
        self.is_solution_is_clique = self.is_clique(self.get_solution_nodes(self.best_solution))

    @staticmethod
    def get_complement_edges(subgraph: nx.Graph) -> List[int]:
        graph_complement = nx.complement(subgraph)
        return list(filter(lambda edge: edge[0] != edge[1], graph_complement.edges()))

    def check_solution(self, curr_values: List[float]) -> Optional[List[int]]:
        solution_nodes = self.get_solution_nodes(curr_values)
        is_clique = self.is_clique(solution_nodes)
        subgraph = self.graph.graph.subgraph(solution_nodes)
        return None if is_clique else self.get_complement_edges(subgraph)

    def left_branching(self, branching_var: Tuple[int, float], cur_branch: int) -> None:
        self.add_left_constraint(branching_var, cur_branch)
        self.branch_and_cut()
        self.cplex_model.linear_constraints.delete(f'c{cur_branch}')

    def right_branching(self, branching_var: Tuple[int, float], cur_branch: int) -> None:
        self.add_right_constraint(branching_var, cur_branch)
        self.branch_and_cut()
        self.cplex_model.linear_constraints.delete(f'c{cur_branch}')

    def branch_and_cut(self) -> None:
        current_objective_value, current_values = self.get_solution()
        if current_objective_value is None or not self.current_solution_is_best(current_objective_value):
            return

        start_time = time.time()
        while time.time() - start_time <= self.tailing_off_time_threshold:
            new_constraints = self.separation(current_values)
            if new_constraints is None:
                break
            self.add_multiple_constraints(new_constraints)
            current_objective_value, current_values = self.get_solution()
            if current_objective_value is None:
                return
            if not self.current_solution_is_best(current_objective_value):
                return

        self.branch_num += 1
        cur_branch = self.branch_num
        branching_var = self.get_branching_var(current_values)
        if branching_var[0] == -1:
            broken_constraints = self.check_solution(current_values)
            if broken_constraints is not None:
                self.add_multiple_constraints(broken_constraints)
                self.branch_and_cut()
            else:
                self.best_solution = [round(x) for x in current_values]
                self.maximum_clique_size = math.floor(current_objective_value)
                self.show_update_solution(self.maximum_clique_size)
            return

        # go to right branch if value closer to 1
        if round(branching_var[1]):
            self.right_branching(branching_var, cur_branch)
            self.left_branching(branching_var, cur_branch)
        else:
            self.left_branching(branching_var, cur_branch)
            self.right_branching(branching_var, cur_branch)
