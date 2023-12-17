import time
from pathlib import Path

from tap import Tap

from algorithms.branch_and_bound import BNBSolver
from algorithms.branch_and_cut import BNCSolver
from algorithms.types import SolverTypes
from algorithms.utils import (ExperiemntResult, ExperimentData, ReportData,
                              read_experiemnt_config)
from graph import Graph


class ArgumentParser(Tap):
    solver: SolverTypes = SolverTypes.BNB
    benchmark_data_path: Path
    experiment_config: Path
    output_results_dump: Path


def benchmark(experiment: ExperimentData, solver_type: SolverTypes, benchmark_data_path: Path) -> ExperiemntResult:
    graph = Graph(experiment, benchmark_data_path)
    graph.independent_sets_generation()
    graph.filter_covered_not_connected()

    solver_map = {
        SolverTypes.BNB: BNBSolver,
        SolverTypes.BNC: BNCSolver
    }
    solver = solver_map[solver_type](graph=graph, debug_mode=True, branching_treshold=1e6)

    start_time = time.time()
    solver.solve()
    end_time = time.time()

    experiemnt_result = ExperiemntResult(
        graph_name=graph.name,
        max_clique=experiment.max_clique,
        founded_clique=solver.maximum_clique_size,
        is_clique=solver.is_solution_is_clique,
        time=end_time - start_time
        )
    return experiemnt_result


def main():
    args = ArgumentParser(underscores_to_dashes=True).parse_args()

    experiments = read_experiemnt_config(args.experiment_config)
    report = ReportData()

    for experiment in experiments:
        # graph_name = experiment.name
        # print(f'Processing the graph {graph_name} with solver {args.solver}', flush=True)
        experiement_result = benchmark(experiment, args.solver, args.benchmark_data_path)
        report.add(experiement_result)
        experiement_result.show()
    report.dump(args.output_results_dump)


if __name__ == "__main__":
    main()
