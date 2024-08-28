from typing import Callable

from docplex.mp.linear import LinearExpr

from src.model.StepVRP import StepVRP
from src.model.VRP import DistanceUnit
from src.solver.cost_functions import manhattan_distance


class MultiCVRP(StepVRP):
    """
    A class to represent a QUBO math formulation of the CVRP model with all vehicles having different capacities.

    This model is always be simplified, since some constraints assume the simplification.

    Attributes:
        capacities (list): List of vehicle capacities.
    """

    def __init__(
        self,
        num_vehicles: int,
        locations: list[tuple[float, float]],
        demands: list[int],
        capacities: list[int],
        cost_function: Callable[
            [list[tuple[float, float]], DistanceUnit], list[list[float]]
        ],
        depot: int | None,
        distance_matrix: list[list[float]] | None,
        location_names: list[str] | None,
        distance_unit: DistanceUnit,
    ):
        self.capacities = capacities
        self.num_locations = len(locations)
        super().__init__(
            num_vehicles,
            locations,
            demands,
            depot,
            cost_function,
            distance_matrix,
            location_names,
            distance_unit,
        )

    def create_objective(self) -> LinearExpr:
        """
        Create the objective function for the CVRP model.
        """

        return self.model.sum(
            self.distance_matrix[i][j]
            / self.normalization_factor
            * self.x_var(k, i, s)
            * self.x_var(k, j, s + 1)
            for k in range(self.num_vehicles)
            for i in range(self.num_used_locations)
            for j in range(self.num_used_locations)
            for s in range(self.num_steps - 1)
        )

    def create_constraints(self):
        """
        Create the constraints for the CVRP model.
        """

        super().create_constraints()
        self.create_capacity_constraints()

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle starts and ends at the depot.
        """

        self.constraints.extend(
            self.model.sum(self.x_var(k, i, s) for i in range(self.num_used_locations))
            == 1
            for k in range(self.num_vehicles)
            for s in range(self.num_steps)
        )

    def create_capacity_constraints(self):
        """
        Create the capacity constraints for the CPLEX model.
        """

        self.constraints.extend(
            self.model.sum(
                self.get_location_demand(i) * self.x_var(k, i, s)
                for i in range(1, self.num_used_locations)
                for s in range(self.num_steps)
            )
            <= self.capacities[k]
            for k in range(self.num_vehicles)
        )

    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that should be replaced during the simplification and their values.
        """

        variables = {}

        for k in range(self.num_vehicles):
            # Vehicles start and end at the depot
            variables[self.get_var_name(k, 0, 0)] = 1
            variables[self.get_var_name(k, 0, self.num_steps - 1)] = 1

            for i in range(1, self.num_used_locations):
                variables[self.get_var_name(k, i, 0)] = 0
                variables[self.get_var_name(k, i, self.num_steps - 1)] = 0

        return variables

    def get_num_steps(self):
        """
        Get the number of steps for the model.
        """

        return len(self.distance_matrix) + 1

    def get_num_used_locations(self):
        """
        Get the number of used locations for the model.
        """

        return self.num_locations

    def get_result_location(
        self, var_dict: dict[str, float], k: int, s: int
    ) -> int | None:
        """
        Get the location for a vehicle at a given step.
        """

        for i in range(self.num_used_locations):
            if self.get_var(var_dict, k, i, s) == 1.0:
                return i

        return None
