from abc import ABC, abstractmethod

from docplex.mp.constr import LinearConstraint
from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model

from src.model.VRP import VRP


class QuboVRP(VRP, ABC):
    """
    A class to represent a QUBO math formulation of the VRP model.
    It uses the utilities from docplex, but it should be adapted to the solver framework used.

    Attributes:
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
        self.model = Model("VRP")
        self.x: list[Var] = []
        self.u: list[Var] = []
        self.constraints: list[LinearConstraint] = []

        self.create_vars()
        self.objective = self.create_objective()
        self.create_constraints()

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

    @abstractmethod
    def x_var(self, i: int, j: int, k: int | None) -> Var:
        """
        Get the value of a variable for the given indices.
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
