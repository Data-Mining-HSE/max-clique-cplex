from enum import Enum


class SolverTypes(Enum):
    BNB = 'bnb'

    def __str__(self) -> str:
        return self.value
