from abc import ABC, abstractmethod

from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import OptimizationResult
from qiskit_optimization.translators import from_docplex_mp

from src.model.VRP import VRP


class CplexVRP(ABC, VRP):
    """
    A class to represent a CPLEX math formulation of the CVRP model.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        locations (list): List of coordinates for each location.
        use_deliveries (bool): Whether the problem uses deliveries or not.
        cplex (Model): CPLEX model for the CVRP
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        use_deliveries: bool,
        simplify: bool,
    ):
        super().__init__(
            num_vehicles, trips, distance_matrix, locations, use_deliveries
        )

        self.simplify = simplify
        self.cplex = Model("CVRP")
        self.build_cplex()

    def build_cplex(self):
        """
        Builds the CPLEX model for CVRP.
        """

        self.create_vars()
        self.create_objective()
        self.create_constraints()

    def quadratic_program(self) -> QuadraticProgram:
        """
        Builds the quadratic program for CVRP, based on CPLEX.
        """

        qp = from_docplex_mp(self.cplex)
        if self.simplify:
            qp = self.simplify_problem(qp)
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
    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that should be replaced during the simplification and their values.
        """
        pass

    def simplify_problem(self, qp: QuadraticProgram) -> QuadraticProgram:
        """
        Simplify the problem by removing unnecessary variables.
        """
        return qp.substitute_variables(self.get_simplified_variables())

    def re_add_variables(self, var_dict: dict[str, float]) -> dict[str, float]:
        """
        Re-add the variables that were removed during the simplification.
        """
        replaced_vars = self.get_simplified_variables()
        for var_name, value in replaced_vars.items():
            var_dict[var_name] = float(value)
        return var_dict

    def build_var_dict(self, result: OptimizationResult) -> dict[str, float]:
        """
        Build a dictionary with the variable values from the result. It takes the simplification step into consideration
        """
        var_dict = result.variables_dict
        if self.simplify:
            var_dict = self.re_add_variables(var_dict)
        return var_dict

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

    @abstractmethod
    def get_var_name(self, i: int, j: int, k: int | None) -> str:
        """
        Get the variable name for the given indices.
        """
        pass

    def get_var(
        self, var_dict: dict[str, float], i: int, j: int, k: int | None = None
    ) -> float:
        """
        Get the variable value for the given indices.
        """

        var_name = self.get_var_name(i, j, k)
        return var_dict[var_name]
