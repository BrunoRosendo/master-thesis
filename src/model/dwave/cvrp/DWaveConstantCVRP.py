import dimod
import numpy as np

from src.model.dwave.DWaveVRP import DWaveVRP


class DWaveConstantCVRP(DWaveVRP):
    """
    A class to represent a DWave Ocean formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        locations (list): List of coordinates for each location.
        use_deliveries (bool): Whether the problem uses deliveries or not.
        capacity (int): Capacity of each vehicle.
        cqm (ConstrainedQuadraticModel): DWave Ocean model for the CVRP
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
    """

    def __init__(
        self,
        num_vehicles: int,
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        simplify: bool,
        capacity: int | None,
    ):
        self.capacity = capacity
        self.copy_vars = False
        super().__init__(num_vehicles, [], distance_matrix, locations, False, simplify)

    def create_vars(self):
        """
        Create the variables for the CQM DWave model.
        """

        self.x = dimod.BinaryArray(
            [
                f"x_{i}_{j}"
                for i in range(self.num_locations)
                for j in range(self.num_locations)
            ]
        )

        self.u = np.array(
            [
                dimod.Integer(
                    f"u_{i}",
                    lower_bound=self.get_u_lower_bound(i),
                    upper_bound=self.get_u_upper_bound(),
                )
                for i in range(1, self.num_locations)
            ],
            dtype=object,
        )

    def create_objective(self):
        """
        Create the objective function for the CQM DWave model.
        """

        self.cqm.set_objective(
            dimod.quicksum(
                self.distance_matrix[i][j] * self.x[i * self.num_locations + j]
                for i in range(self.num_locations)
                for j in range(self.num_locations)
            )
        )

    def create_constraints(self):
        """
        Create the constraints for the CQM DWave model.
        """
        self.create_location_constraints()
        self.create_vehicle_constraints()
        self.create_subtour_constraints()

    def create_location_constraints(self):
        """
        Create the constraints that ensure each location is visited exactly once.
        """

        for i in range(1, self.num_locations):
            self.cqm.add_constraint(
                dimod.quicksum(
                    self.x[i * self.num_locations + j]
                    for j in range(self.num_locations)
                    if i != j
                )
                == 1,
                copy=self.copy_vars,
            )
            self.cqm.add_constraint(
                dimod.quicksum(
                    self.x[j * self.num_locations + i]
                    for j in range(self.num_locations)
                    if i != j
                )
                == 1,
                copy=self.copy_vars,
            )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle starts and ends at the depot.
        """

        self.cqm.add_constraint(
            dimod.quicksum(self.x[i] for i in range(1, self.num_locations))
            == self.num_vehicles,
            copy=self.copy_vars,
        )

        self.cqm.add_constraint(
            dimod.quicksum(
                self.x[i * self.num_locations] for i in range(1, self.num_locations)
            )
            == self.num_vehicles,
            copy=self.copy_vars,
        )

    def create_subtour_constraints(self):
        """
        Create the constraints that eliminate subtours (MTV).
        """

        for i in range(1, self.num_locations):
            for j in range(1, self.num_locations):
                if i != j:
                    self.cqm.add_constraint(
                        self.u[i - 1]
                        - self.u[j - 1]
                        + self.capacity * self.x[i * self.num_locations + j]
                        <= self.capacity - self.get_location_demand(j)
                    )

    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that are relevant for the solution.
        """

        return {self.get_var_name(i, i): 0 for i in range(len(self.distance_matrix))}

    def get_var_name(self, i: int, j: int, k: int | None = None) -> str:
        """
        Get the name of a variable.
        """

        return f"x_{i}_{j}"

    def get_u_lower_bound(self, i: int) -> int:
        """
        Get the lower bound for the u variable, at the given index.
        """

        return self.get_location_demand(i)

    def get_u_upper_bound(self) -> int:
        """
        Get the upper bound for the u variable.
        """

        return self.capacity
