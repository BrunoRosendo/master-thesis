from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr

from src.model.VRPSolution import DistanceUnit
from src.model.qubo.QuboVRP import QuboVRP


class ConstantCVRP(QuboVRP):
    """
    A class to represent a QUBO math formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        capacity (int): Capacity of each vehicle.
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[float]],
        capacity: int | None,
        locations: list[tuple[float, float]],
        simplify: bool,
        location_names: list[str] = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
    ):
        self.capacity = capacity
        super().__init__(
            num_vehicles,
            trips,
            distance_matrix,
            locations,
            False,
            simplify,
            location_names=location_names,
            distance_unit=distance_unit,
        )

    def create_vars(self):
        """
        Create the variables for the CVRP model.
        """

        self.x.extend(
            self.model.binary_var(self.get_var_name(i, j))
            for i in range(self.num_locations)
            for j in range(self.num_locations)
        )

        self.u.extend(
            self.model.integer_var(
                self.get_u_lower_bound(i), self.get_u_upper_bound(), f"u_{i}"
            )
            for i in range(1, self.num_locations)
        )

    def create_objective(self) -> LinearExpr:
        """
        Create the objective function for the CVRP model.
        """

        return self.model.sum(
            self.distance_matrix[i][j] * self.x_var(i, j)
            for i in range(self.num_locations)
            for j in range(self.num_locations)
        )

    def create_constraints(self):
        """
        Create the constraints for the CVRP model.
        """

        self.create_location_constraints()
        self.create_vehicle_constraints()
        self.create_subtour_constraints()

    def create_location_constraints(self):
        """
        Create the constraints that ensure each location is visited exactly once.
        """

        self.constraints.extend(
            self.model.sum(
                self.x_var(i, j) for j in range(self.num_locations) if i != j
            )
            == 1
            for i in range(1, self.num_locations)
        )

        self.constraints.extend(
            self.model.sum(
                self.x_var(j, i) for j in range(self.num_locations) if i != j
            )
            == 1
            for i in range(1, self.num_locations)
        )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle starts and ends at the depot.
        """

        self.constraints.append(
            self.model.sum(self.x_var(0, i) for i in range(1, self.num_locations))
            == self.num_vehicles,
        )

        self.constraints.append(
            self.model.sum(self.x_var(i, 0) for i in range(1, self.num_locations))
            == self.num_vehicles,
        )

    def create_subtour_constraints(self):
        """
        Create the constraints that eliminate subtours (MTV).
        """

        self.constraints.extend(
            self.u[i - 1] - self.u[j - 1] + self.capacity * self.x_var(i, j)
            <= self.capacity - self.get_location_demand(j)
            for i in range(1, self.num_locations)
            for j in range(1, self.num_locations)
            if i != j
        )

    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that are relevant for the solution.
        """

        return {self.get_var_name(i, i): 0 for i in range(len(self.distance_matrix))}

    def get_result_route_starts(self, var_dict: dict[str, float]) -> list[int]:
        """
        Get the starting location for each route from the variable dictionary.
        """
        route_starts = []

        cur_location = 1
        while len(route_starts) < self.num_vehicles:
            var_value = self.get_var(var_dict, 0, cur_location)
            if var_value == 1.0:
                route_starts.append(cur_location)
            cur_location += 1

        return route_starts

    def get_result_next_location(
        self, var_dict: dict[str, float], cur_location: int
    ) -> int | None:
        """
        Get the next location for a route from the variable dictionary.
        """

        for i in range(len(self.locations)):
            var_value = self.get_var(var_dict, cur_location, i)
            if var_value == 1.0:
                return i
        return None

    def get_var_name(self, i: int, j: int, k: int | None = None) -> str:
        """
        Get the name of a decision variable.
        """

        return f"x_{i}_{j}"

    def x_var(self, i: int, j: int, k: int | None = None) -> Var:
        return self.x[i * self.num_locations + j]

    def get_u_lower_bound(self, i: int) -> int:
        """
        Get the lower bound for the auxiliary variable, at the given index.
        """

        return self.get_location_demand(i)

    def get_u_upper_bound(self) -> int:
        """
        Get the upper bound for the auxiliary variables.
        """

        return self.capacity

    def get_capacity(self) -> int | list[int] | None:
        """
        Get the capacity of the vehicles.
        """
        return self.capacity
