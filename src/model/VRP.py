from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable

from docplex.mp.constr import LinearConstraint
from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model


class DistanceUnit(str, Enum):
    METERS = "METERS"
    SECONDS = "SECONDS"


class VRP(ABC):
    """
    A class to represent a QUBO math formulation of the VRP model.
    It uses the utilities from docplex, but it should be adapted to the solver framework used.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        num_locations (int): Number of locations.
        locations (list): List of coordinates for each location.
        demands (list): List of demands for each location.
        depot (int | None): Index of the depot location, if it exists.
        location_names (list): List of names for each location. Optional.
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
        model (Model): Dummy docplex model for the VRP.
        x (list): List of binary decision variables for the VRP model.
        u (list): List of integer auxiliary variables for the VRP model.
        constraints (list): List of constraints for the VRP model.
        objective (LinearExpr): Objective function for the VRP model.
    """

    def __init__(
        self,
        num_vehicles: int,
        locations: list[tuple[float, float]],
        demands: list[int] | None,
        depot: int | None,
        simplify: bool,
        cost_function: Callable[
            [list[tuple[float, float]], DistanceUnit], list[list[float]]
        ],
        distance_matrix: list[list[float]] | None,
        location_names: list[str] | None,
        distance_unit: DistanceUnit,
    ):
        self.num_vehicles = num_vehicles
        self.locations = locations
        self.demands = demands
        self.depot = depot
        self.simplify = simplify
        self.cost_function = cost_function
        self.distance_matrix = distance_matrix
        self.location_names = location_names
        self.distance_unit = distance_unit

        self.model = Model("VRP")
        self.x: list[Var] = []
        self.u: list[Var] = []
        self.constraints: list[LinearConstraint] = []

        self.create_distance_matrix()
        self.num_locations = len(self.distance_matrix)
        self.create_vars()
        self.objective = self.create_objective()
        self.create_constraints()

    def create_distance_matrix(self):
        """
        Create the distance matrix for the VRP model, if it was not provided.
        """
        if self.distance_matrix is None:
            self.distance_matrix = self.cost_function(
                self.locations, self.distance_unit
            )

    def get_location_demand(self, idx: int) -> int:
        """
        Get the demand for a location.
        """
        if idx == self.depot:
            return 0

        return max(self.demands[idx], 1)

    @abstractmethod
    def create_vars(self):
        """
        Create the variables for the VRP model.
        """
        pass

    @abstractmethod
    def create_objective(self) -> LinearExpr:
        """
        Create the objective function for the VRP model.
        """
        pass

    @abstractmethod
    def create_constraints(self):
        """
        Create the constraints for the VRP model.
        """
        pass

    @abstractmethod
    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that should be replaced during the simplification and their values.
        """
        pass

    def re_add_variables(self, var_dict: dict[str, float]) -> dict[str, float]:
        """
        Re-add the variables that were removed during the simplification.
        """
        replaced_vars = self.get_simplified_variables()
        for var_name, value in replaced_vars.items():
            var_dict[var_name] = float(value)
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

    def is_result_feasible(self, var_dict: dict[str, float]) -> bool:
        """
        Check if the result is feasible. This method can optionally be implemented by the subclass.
        """
        return True

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
        return round(var_dict[var_name])

    @abstractmethod
    def x_var(self, i: int, j: int, k: int | None) -> Var:
        """
        Get the value of a variable for the given indices.
        """
        pass
