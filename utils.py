import csv
import datetime
import os
import os.path as osp
import time
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import List

import networkx as nx
from loguru import logger

DATA_DIR = osp.join(osp.dirname(__file__), "benchmarks")
SOURCE_GRAPH_DIR = osp.join(osp.dirname(__file__), "data")
RESULTS_DIR = osp.join(osp.dirname(__file__), "results")
LOG_DIR = osp.join(osp.dirname(__file__), "becnhmark_logs")

EPS = 1e-5
STRATEGIES = [
    nx.coloring.strategy_largest_first,
    nx.coloring.strategy_random_sequential,
    nx.coloring.strategy_connected_sequential_bfs,
    nx.coloring.strategy_connected_sequential_dfs,
    nx.coloring.strategy_saturation_largest_first,
    nx.coloring.strategy_smallest_last,
]

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")


def timeit(f):
    """Measures time of function execution"""

    def wrap(*args):
        time1 = time.time()
        result = f(*args)
        time2 = time.time()
        work_time = round(time2 - time1, 3)
        logger.info(f"Function: <{f.__name__}> worked {work_time} seconds")
        return result, work_time

    return wrap

@dataclass
class ExperimentData:
    name: str
    max_clique: int

@dataclass
class ExperiemntResult:
    """
    Storing the result of experiemnt
    """
    graph_name: str
    max_clique: int
    founded_clique: int
    is_clique: bool
    time: int

    def show(self) -> None:
        msg = f'Graph: {self.graph_name}. Clique is found: {self.is_clique}) with size {self.founded_clique}. Time {self.time}. Max known clique {self.max_clique}'
        print(msg, flush=True)


class ReportData:
    """
    Collect all results for processing report
    """
    def __init__(self) -> None:
        self.graph_names: List[str] = []
        self.max_cliques: List[int] = []
        self.founded_cliques: List[int] = []
        self.is_cliques: List[bool] = []
        self.times: List[int] = []

    @property
    def header(self) -> List[str]:
        column_names = [
            'Graph Name',
            'Correct Max Clique',
            'Found Max Clique',
            'Is Clique',
            'Consumed Time',
        ]
        return column_names
    
    def add(self, result: ExperiemntResult) -> None:
        self.graph_names.append(result.graph_name)
        self.max_cliques.append(result.max_clique)
        self.founded_cliques.append(result.founded_clique)
        self.is_cliques.append(result.is_clique)
        self.times.append(result.time)

    def dump(self, output_path: Path) -> None:
        output_path.mkdir(exist_ok=True)
        # with output_path.open('w') as file_buff:
        #     writer = csv.writer(file_buff)
        #     writer.writerows(results)

def read_experiemnt_config(experiment_config: Path) -> List[ExperimentData]:
    """
    This function returns the list of ExperimentData objects
    with information of graphs in experiment list: Name of graph and Max Clique
    """
    with experiment_config.open() as test_data:
        experiments = [ExperimentData(*tuple(line.strip().split(','))) for line in test_data.readlines()]
        return experiments
