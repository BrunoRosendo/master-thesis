from abc import ABC, abstractmethod

from src.model.qubo.QuboVRP import QuboVRP


class Adapter(ABC):
    """
    A class to represent an adapter for the QUBO model.
    It should be implemented by the solver-specific adapter.

    Attributes:
        qubo (QuboVRP): QUBO model to be adapted.
    """

    def __init__(self, qubo: QuboVRP):
        self.qubo = qubo

        self.add_vars()
        self.add_objective()
        self.add_constraints()

    @abstractmethod
    def add_vars(self):
        """
        Add the QUBO variables to the model used.
        """
        pass

    @abstractmethod
    def add_objective(self):
        """
        Add the QUBO objective function to the model used.
        """
        pass

    @abstractmethod
    def add_constraints(self):
        """
        Add the QUBO constraints to the model used.
        """
        pass

    @abstractmethod
    def get_model(self) -> object:
        """
        Returns the model for VRP, based on the solver.
        """
        pass
