import numpy as np
from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr

from src.model.qubo.QuboVRP import QuboVRP


class MultiCVRP(QuboVRP):
    """
    A class to represent a QUBO math formulation of the CVRP model with all vehicles having different capacities.

    This model is always be simplified, since some constraints assume the simplification.

    Attributes:
        capacities (list): List of vehicle capacities.
    """

    def __init__(
        self,
        num_vehicles: int,
        distance_matrix: list[list[int]],
        capacities: list[int],
        locations: list[tuple[int, int]],
    ):
        self.capacities = capacities
        self.num_steps = len(distance_matrix) + 1

        self.epsilon = 0  # TODO Check this value
        self.normalization_factor = np.max(distance_matrix) + self.epsilon

        self.copy_vars = False

        super().__init__(num_vehicles, [], distance_matrix, locations, False, True)

    def create_vars(self):
        """
        Create the variables for the CVRP model.
        """

        self.x.extend(
            self.model.binary_var(self.get_var_name(k, i, s))
            for k in range(self.num_vehicles)
            for i in range(self.num_locations)
            for s in range(self.num_steps)
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
            for i in range(self.num_locations)
            for j in range(self.num_locations)
            for s in range(self.num_steps - 1)
        )

    def create_constraints(self):
        """
        Create the constraints for the CPLEX model.
        """

        self.create_location_constraints()
        self.create_vehicle_constraints()
        self.create_capacity_constraints()

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
            for i in range(1, self.num_locations)
        )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle starts and ends at the depot.
        """

        self.constraints.extend(
            self.model.sum(self.x_var(k, i, s) for i in range(self.num_locations)) == 1
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
                for i in range(1, self.num_locations)
                for s in range(cur_step + 1)
            )
            <= self.capacities[k]
            for k in range(self.num_vehicles)
            for cur_step in range(1, self.num_steps)
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

            for i in range(1, self.num_locations):
                variables[self.get_var_name(k, i, 0)] = 0
                variables[self.get_var_name(k, i, self.num_steps - 1)] = 0

        return variables

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

    def get_result_location(
        self, var_dict: dict[str, float], k: int, s: int
    ) -> int | None:
        """
        Get the location for a vehicle at a given step.
        """

        for i in range(self.num_locations):
            if self.get_var(var_dict, k, i, s) == 1.0:
                return i

        return None

    def get_var_name(self, k: int, i: int, s: int | None = None) -> str:
        """
        Get the name of a variable.
        """

        return f"x_{k}_{i}_{s}"

    def x_var(self, k: int, i: int, s: int) -> Var:
        return self.x[k * self.num_locations * self.num_steps + i * self.num_steps + s]
