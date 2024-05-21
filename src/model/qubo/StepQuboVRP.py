from abc import ABC, abstractmethod

import numpy as np
from docplex.mp.dvar import Var

from src.model.qubo.QuboVRP import QuboVRP


class StepQuboVRP(QuboVRP, ABC):
    """
    A class to represent a QUBO math formulation of the step-based CVRP model.
    This model should always be simplified, since some constraints assume the simplification.

    Attributes:
        num_steps (int): Number of max steps for each vehicle.
        num_used_locations (int): Number of locations used in the problem, including the auxiliary depot if used.
        epsilon (int): Small value to avoid division by zero.
        normalization_factor (int): Value to normalize the objective function.
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[float]],
        locations: list[tuple[int, int]],
        use_deliveries: bool,
        depot: int | None = 0,
        location_names: list[str] = None,
    ):
        self.distance_matrix = distance_matrix
        self.num_steps = self.get_num_steps()
        self.num_used_locations = self.get_num_used_locations()
        self.epsilon = 0.0001
        self.normalization_factor = np.max(self.distance_matrix) + self.epsilon

        super().__init__(
            num_vehicles,
            trips,
            distance_matrix,
            locations,
            use_deliveries,
            True,
            depot,
            location_names,
        )

    def create_vars(self):
        """
        Create the variables for the VRP model.
        """

        self.x.extend(
            self.model.binary_var(self.get_var_name(k, i, s))
            for k in range(self.num_vehicles)
            for i in range(self.num_used_locations)
            for s in range(self.num_steps)
        )

    def create_constraints(self):
        """
        Create the constraints for the VRP model.
        """

        self.create_location_constraints()
        self.create_vehicle_constraints()

    def create_location_constraints(self):
        """
        Create the constraints that ensure each location is visited exactly once.
        """

        self.constraints.extend(
            self.model.sum(
                self.x_var(k, i, s)
                for k in range(self.num_vehicles)
                for s in range(self.num_steps)
            )
            == 1
            for i in range(1, self.num_used_locations)
        )

    @abstractmethod
    def create_vehicle_constraints(self):
        """
        Create the vehicle constraints, depending on the model.
        """
        pass

    @abstractmethod
    def get_num_steps(self):
        """
        Get the number of steps for the model.
        """
        pass

    @abstractmethod
    def get_num_used_locations(self):
        """
        Get the number of used locations for the model.
        """
        pass

    def get_result_route_starts(self, var_dict: dict[str, float]) -> list[int]:
        """
        Get the starting location for each route from the variable dictionary.
        """

        route_starts = []

        for k in range(self.num_vehicles):
            for s in range(self.num_steps):
                if self.get_var(var_dict, k, 0, s) == 0.0:
                    start = self.get_result_location(var_dict, k, s)
                    route_starts.append(start)
                    break

        return route_starts

    def get_result_next_location(
        self, var_dict: dict[str, float], cur_location: int
    ) -> int | None:
        """
        Get the next location for a route from the variable dictionary.
        """
        for k in range(self.num_vehicles):
            for s in range(self.num_steps - 1):
                if self.get_var(var_dict, k, cur_location, s) == 1.0:
                    return self.get_result_location(var_dict, k, s + 1)

        return None

    @abstractmethod
    def get_result_location(
        self, var_dict: dict[str, float], k: int, s: int
    ) -> int | None:
        """
        Get the location for a vehicle at a given step.
        """
        pass

    def get_var_name(self, k: int, i: int, s: int | None = None) -> str:
        """
        Get the name of a variable.
        """

        return f"x_{k}_{i}_{s}"

    def x_var(self, k: int, i: int, s: int) -> Var:
        return self.x[
            k * self.num_used_locations * self.num_steps + i * self.num_steps + s
        ]
