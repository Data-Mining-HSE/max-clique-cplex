from math import inf
from typing import Union

import numpy as np
from numpy.typing import NDArray

from utils import *


def read_dimacs_adjacency_matrix(path: Path) -> NDArray[np.int8]:
    with path.open() as file_buff:
        for line in file_buff:
            if line.startswith('p'):
                _, _, vertices_num, _ = line.split()
                adjacency_matrix = np.zeros((int(vertices_num), int(vertices_num)), dtype=np.int8)
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                adjacency_matrix[int(v1) - 1][int(v2) - 1] = 1
    return adjacency_matrix


class MCPGraph:
    def __init__(self, experiment: ExperimentData, benchmark_data_path: Path):
        self.name = experiment.name
        self.graph_path = benchmark_data_path / f'{self.name}.clq'

        adj_matrix = read_dimacs_adjacency_matrix(self.graph_path)
        self.graph = nx.from_numpy_array(adj_matrix)
        self.maximum_clique_size_gt = experiment.max_clique

        self.maximum_clique_size_found = -1
        self.independent_vertex_sets = set()
        self.not_connected_vertexes = nx.complement(self.graph).edges
        self.nodes = self.graph.nodes
        self.is_solution_is_clique = None

    def independent_sets_generation(
        self,
        minimum_set_size: int = 3,
        iteration_number: int = 50,
        time_limit: int = 500,
        max_weighted: bool = False,
        solution=None,
        strategies=STRATEGIES,
    ):
        """Independent Vertex Sets generation via graph coloring

            This function is also solve Max Weighted Independent Sets problem (Not a proper way of solve
                                                                                            it via coloring. I know)

        Returns:
        Nothing returns. Function update self.independent_vertex_sets field if max_weighted = False
                                                                            else return set of weighted ind sets
        """
        generated_independent_sets = (
            self.independent_vertex_sets if not max_weighted else set()
        )
        if len(self.graph.nodes) < 500 and not max_weighted:
            strategies.append(
                nx.coloring.strategy_independent_set,
            )
        start_time = time.time()
        for _ in range(iteration_number):
            if time.time() - start_time >= time_limit:
                logger.info("Reach time limit at searching ind sets")
                break

            for strategy in strategies:
                dict_of_independet_sets = dict()
                # get coloring with current strategy: running_coloring - dict(key=vertex, value=color)
                running_coloring = nx.coloring.greedy_color(
                    self.graph,
                    strategy=strategy,
                )
                for vertex, color in running_coloring.items():
                    if color not in dict_of_independet_sets.keys():
                        dict_of_independet_sets[color] = []

                    if not max_weighted:
                        dict_of_independet_sets[color].append(vertex)
                    else:
                        dict_of_independet_sets[color].append(
                            (vertex, solution[vertex]),
                        )

                for _, ind_set in dict_of_independet_sets.items():
                    set_weight = (
                        sum(vertex[1] for vertex in ind_set)
                        if max_weighted
                        else inf
                    )
                    ind_set = (
                        [vertex[0] for vertex in ind_set]
                        if max_weighted
                        else ind_set
                    )
                    if max_weighted:
                        if (
                            len(ind_set) >= minimum_set_size
                            and set_weight > 1 + EPS
                        ):
                            generated_independent_sets.add(
                                tuple((tuple(ind_set), set_weight)),
                            )
                    else:
                        if len(ind_set) >= minimum_set_size:
                            generated_independent_sets.add(
                                tuple(sorted(ind_set)),
                            )
        if max_weighted:
            return generated_independent_sets

    def filter_covered_not_connected(self, filtration_limit: int = 300000):
        filtered_not_connected = []
        for idx, not_connected_vertexes in enumerate(
            self.not_connected_vertexes,
        ):
            vertexes_are_covered_by_set = False
            vertex_1, vertex_2 = not_connected_vertexes
            if idx < filtration_limit:
                for ind_set in self.independent_vertex_sets:
                    if (vertex_1 in ind_set) and (vertex_2 in ind_set):
                        vertexes_are_covered_by_set = True
                        break
            if not vertexes_are_covered_by_set:
                filtered_not_connected.append(not_connected_vertexes)
        self.not_connected_vertexes = filtered_not_connected
