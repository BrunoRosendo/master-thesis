from abc import ABC, abstractmethod

from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp

from src.model.CVRPModel import CVRPModel


class CPLEXModel(ABC, CVRPModel):
    """
    A class to represent a CPLEX math formulation of the CVRP model.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        cplex (Model): CPLEX model for the CVRP
    """

    def __init__(self, num_vehicles, trips, depot, distance_matrix, locations):
        super().__init__(num_vehicles, trips, depot, distance_matrix, locations)

        self.cplex = Model("CVRP")
        self.build_cplex()

    def build_cplex(self):
        """
        Builds the CPLEX model for CVRP.
        """

        self.create_vars()
        self.create_objective()
        self.create_constraints()

    def quadratic_program(self, simplify=True) -> QuadraticProgram:
        """
        Builds the quadratic program for CVRP, based on CPLEX.
        """

        qp = from_docplex_mp(self.cplex)
        if simplify:
            qp = self.simplify(qp)
        return qp

    @abstractmethod
    def create_vars(self):
        """
        Create the variables for the CPLEX model.
        """
        pass

    @abstractmethod
    def create_objective(self):
        """
        Create the objective function for the CPLEX model.
        """
        pass

    @abstractmethod
    def create_constraints(self):
        """
        Create the constraints for the CPLEX model.
        """
        pass

    @abstractmethod
    def simplify(self, qp: QuadraticProgram) -> QuadraticProgram:
        """
        Simplify the problem by removing unnecessary variables.
        """
        pass

    @abstractmethod
    def get_result_route_starts(self, var_dict: dict[str, float]) -> list[int]:
        """
        Get the starting location for each route from the variable dictionary.
        """
        pass

    @abstractmethod
    def get_result_next_location(
        self, var_dict: dict[str, float], cur_location: int
    ) -> int | None:
        """
        Get the next location for a route from the variable dictionary.
        """
        pass
