import time
from enum import Enum
from pathlib import Path

from tap import Tap

from algorithms.branch_and_bound import BNBSolver
from algorithms.branch_and_cut import BNCSolver
from graph import MCPGraph
from utils import *


class SolverTypes(Enum):
    BNB = 'bnb'
    BNC = 'bnc'

    def __str__(self) -> str:
        return self.value


class ArgumentParser(Tap):
    solver: SolverTypes = SolverTypes.BNB
    experiment_config: Path = Path('test.txt')
    output_results_dump: Path = Path('results.csv')
    benchmark_data_path: Path


def benchmark(experiment: ExperimentData, solver_type: SolverTypes, benchmark_data_path: Path) -> ExperiemntResult:
    graph = MCPGraph(experiment, benchmark_data_path)
    graph.independent_sets_generation()
    graph.filter_covered_not_connected()

    if solver_type is SolverTypes.BNB:
        solver_initer = BNBSolver
    elif solver_type is SolverTypes.BNC:
        solver_initer = BNCSolver
    
    solver = solver_initer(graph=graph)

    start_time = time.time()
    solver.solve()
    end_time = time.time()

    graph.maximum_clique_size_found = solver.maximum_clique_size
    graph.is_solution_is_clique = solver.is_solution_is_clique

    experiemnt_result = ExperiemntResult(
        graph_name=graph.name,
        max_clique=graph.maximum_clique_size_gt,
        founded_clique=solver.maximum_clique_size,
        is_clique=solver.is_solution_is_clique[0],
        time=end_time - start_time
        )
    return experiemnt_result


def main():
    args = ArgumentParser(underscores_to_dashes=True).parse_args()

    experiments = read_experiemnt_config(args.experiment_config)

    report = ReportData()

    for experiment in experiments:
        graph_name = experiment.name
        print(f'Processing the graph {graph_name} with solver {args.solver}')

        experiement_result = benchmark(experiment, args.solver, args.benchmark_data_path)
        report.add(experiement_result)
        experiement_result.show()

    report.dump(args.output_results_dump)


if __name__ == "__main__":
    main()
