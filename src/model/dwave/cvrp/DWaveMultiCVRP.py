import dimod
import numpy as np

from src.model.dwave.DWaveVRP import DWaveVRP


class DWaveMultiCVRP(DWaveVRP):
    """
    A class to represent a DWave Ocean formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        locations (list): List of coordinates for each location.
        capacities (list): List of vehicle capacities.
        cqm (ConstrainedQuadraticModel): DWave Ocean model for the CVRP
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
        Create the variables for the CQM DWave model.
        """

        self.x = dimod.BinaryArray(
            [
                self.get_var_name(k, i, s)
                for k in range(self.num_vehicles)
                for i in range(self.num_locations)
                for s in range(self.num_steps)
            ]
        )

    def create_objective(self):
        """
        Create the objective function for the CQM DWave model.
        """

        self.cqm.set_objective(
            dimod.quicksum(
                self.distance_matrix[i][j]
                / self.normalization_factor
                * self.x_var(k, i, s)
                * self.x_var(k, j, s + 1)
                for k in range(self.num_vehicles)
                for i in range(self.num_locations)
                for j in range(self.num_locations)
                for s in range(self.num_steps - 1)
            )
        )

    def create_constraints(self):
        """
        Create the constraints for the CQM DWave model.
        """

        self.create_location_constraints()
        self.create_vehicle_constraints()
        self.create_capacity_constraints()

    def create_location_constraints(self):
        """
        Create the constraints that ensure each location is visited exactly once.
        """

        for i in range(1, self.num_locations):
            self.cqm.add_constraint(
                dimod.quicksum(
                    self.x_var(k, i, s)
                    for k in range(self.num_vehicles)
                    for s in range(self.num_steps)
                )
                == 1,
                copy=self.copy_vars,
            )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle starts and ends at the depot.
        """

        for k in range(self.num_vehicles):
            for s in range(self.num_steps):
                self.cqm.add_constraint(
                    dimod.quicksum(
                        self.x_var(k, i, s) for i in range(self.num_locations)
                    )
                    == 1,
                    copy=self.copy_vars,
                )

    def create_capacity_constraints(self):
        """
        Create the capacity constraints for the CPLEX model.
        """

        for k in range(self.num_vehicles):
            for cur_step in range(1, self.num_steps):  # depot has no demand
                self.cqm.add_constraint(
                    dimod.quicksum(
                        self.get_location_demand(i) * self.x_var(k, i, s)
                        for i in range(1, self.num_locations)
                        for s in range(cur_step + 1)
                    )
                    <= self.capacities[k],
                    copy=self.copy_vars,
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

    def x_var(self, k: int, i: int, s: int) -> int:
        return self.x[k * self.num_locations * self.num_steps + i * self.num_steps + s]

    def get_var_name(self, k: int, i: int, s: int | None = None) -> str:
        """
        Get the name of a variable.
        """

        return f"x_{k}_{i}_{s}"
