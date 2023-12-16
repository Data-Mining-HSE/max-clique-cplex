import math

import cplex
import numpy as np

from algorithms.base import MaxCliqueSolver
from graph import Graph


class BNBSolver(MaxCliqueSolver):
    def __init__(self, graph: Graph):
        super().__init__(graph=graph)
        self.cplex_model = self.construct_model()

    def construct_model(self):
        nodes_amount = len(self.graph.nodes)
        obj = [1.0] * nodes_amount
        upper_bounds = [1.0] * nodes_amount
        lower_bounds = [0.0] * nodes_amount
        types = ['C'] * nodes_amount
        columns_names = [f'x{x}' for x in range(nodes_amount)]

        not_connected_edges_amount = len(self.graph.not_connected_vertexes)
        independent_vertex_sets_amount = len(self.graph.independent_vertex_sets)

        right_hand_side = [1.0] * (not_connected_edges_amount + independent_vertex_sets_amount)
        constraint_names = [f'c{x}' for x in range(not_connected_edges_amount + independent_vertex_sets_amount)]
        constraint_senses = ['L'] * (not_connected_edges_amount + independent_vertex_sets_amount)

        problem = cplex.Cplex()
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.set_error_stream(None)
        problem.objective.set_sense(problem.objective.sense.maximize)

        problem.variables.add(
            obj=obj,
            ub=upper_bounds,
            lb=lower_bounds,
            names=columns_names,
            types=types,
        )

        constraints = []
        # set constraints for all vertexes in independent set x_0 + x_1 + ... +  x_i  <=1 with i = len(independent_set)
        for ind_set in self.graph.independent_vertex_sets:
            constraint = [[f'x{i}' for i in ind_set], [1.0] * len(ind_set)]
            constraints.append(constraint)

        # set constraints for not connected edges x_i + x_j <=1
        for xi, xj in self.graph.not_connected_vertexes:
            contraint = [[f'x{xi}', f'x{xj}'], [1.0, 1.0]]
            constraints.append(contraint)

        problem.linear_constraints.add(
            lin_expr=constraints,
            senses=constraint_senses,
            rhs=right_hand_side,
            names=constraint_names,
        )
        return problem

    def solve(self):
        self.init_model_with_heuristic_solution()

        self.branching()
        solution_nodes = np.where(np.isclose(self.best_solution, 1.0, atol=self.eps))

        self.is_solution_is_clique = self.is_clique(solution_nodes[0].tolist())

    def goto_left_branch(self, branching_var, cur_branch):
        self.add_left_constraint(branching_var, cur_branch)
        self.branching()
        self.cplex_model.linear_constraints.delete(f'c{cur_branch}')

    def goto_right_branch(self, branching_var, cur_branch):
        self.add_right_constraint(branching_var, cur_branch)
        self.branching()
        self.cplex_model.linear_constraints.delete(f'c{cur_branch}')

    def branching(self):
        self.cplex_model.solve()

        current_values = self.cplex_model.solution.get_values()
        current_objective_value = self.cplex_model.solution.get_objective_value()

        if not self.current_solution_is_best(current_objective_value):
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
        # go to  right branch if value closer to 1
        if round(branching_var[1]):
            self.goto_right_branch(branching_var, cur_branch)
            self.goto_left_branch(branching_var, cur_branch)
        else:
            self.goto_left_branch(branching_var, cur_branch)
            self.goto_right_branch(branching_var, cur_branch)
