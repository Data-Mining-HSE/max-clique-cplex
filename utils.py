from dataclasses import dataclass
from pathlib import Path
from typing import List


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
        msg = f'Graph: {self.graph_name}. Clique is found: {self.is_clique}. Size {self.founded_clique} (Best {self.max_clique}). Time {self.time}.'
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
