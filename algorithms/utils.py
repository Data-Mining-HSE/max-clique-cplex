from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


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
        msg = f'Graph: {self.graph_name}. Clique is found: {self.is_clique}. Size {self.founded_clique} (Best {self.max_clique}). Time {self.time:.2f}.'
        print(msg, flush=True)


class ReportData:
    def __init__(self) -> None:
        self.graph_names: List[str] = []
        self.max_cliques: List[int] = []
        self.founded_cliques: List[int] = []
        self.is_cliques: List[bool] = []
        self.times: List[int] = []

    @property
    def header(self) -> List[str]:
        column_names = ['Name', 'Max Clique', 'Found', 'Is Clique', 'Time']
        return column_names
    
    def add(self, result: ExperiemntResult) -> None:
        self.graph_names.append(result.graph_name)
        self.max_cliques.append(result.max_clique)
        self.founded_cliques.append(result.founded_clique)
        self.is_cliques.append(result.is_clique)
        self.times.append(result.time)

    def dump(self, output_path: Path) -> None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        data = [self.graph_names, self.max_cliques, self.founded_cliques, self.is_cliques, self.times]
        frame = pd.DataFrame(data, index = self.header).T
        frame.to_csv(output_path, index=False)


def read_experiemnt_config(experiment_config: Path) -> List[ExperimentData]:
    """
    Returns the list of ExperimentData objects with information of graph
    """
    with experiment_config.open() as test_data:
        experiments = [ExperimentData(*tuple(line.strip().split(','))) for line in test_data.readlines()]
        return experiments
