from enum import Enum


class SolverTypes(Enum):
    BNB = 'bnb'
    BNC = 'bnc'

    def __str__(self) -> str:
        return self.value
