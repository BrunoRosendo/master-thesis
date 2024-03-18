from qiskit_optimization import QuadraticProgram

from src.model.cplex.CplexVRP import CplexVRP


class ConstantCVRP(CplexVRP):
    """
    A class to represent a CPLEX math formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        capacity (int): Capacity of each vehicle.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
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
        depot: int,
        distance_matrix: list[list[int]],
        capacity: int | None,
        locations: list[tuple[int, int]],
        use_deliveries: bool,
        simplify: bool,
    ):
        self.capacity = capacity
        super().__init__(
            num_vehicles,
            trips,
            depot,
            distance_matrix,
            locations,
            use_deliveries,
            simplify,
        )

    def create_vars(self):
        """
        Create the variables for the CPLEX model.
        """

        self.x = self.cplex.binary_var_matrix(
            self.num_locations, self.num_locations, name="x"
        )

        self.u = self.cplex.integer_var_list(
            range(1, self.num_locations), name="u", lb=0, ub=self.get_u_upper_bound()
        )

    def create_objective(self):
        """
        Create the objective function for the CPLEX model.
        """

        objective = self.cplex.sum(
            self.distance_matrix[i][j] * self.x[i, j]
            for i in range(self.num_locations)
            for j in range(self.num_locations)
        )
        self.cplex.minimize(objective)

    def create_constraints(self):
        """
        Create the constraints for the CPLEX model.
        """

        self.create_location_constraints()
        self.create_vehicle_constraints()
        self.create_subtour_constraints()

    def create_location_constraints(self):
        """
        Create the constraints that ensure each location is visited exactly once.
        """

        for i in range(1, self.num_locations):
            self.cplex.add_constraint(
                self.cplex.sum(
                    self.x[i, j] for j in range(self.num_locations) if i != j
                )
                == 1
            )
            self.cplex.add_constraint(
                self.cplex.sum(
                    self.x[j, i] for j in range(self.num_locations) if i != j
                )
                == 1
            )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle starts and ends at the depot.
        """

        self.cplex.add_constraint(
            self.cplex.sum(self.x[0, i] for i in range(1, self.num_locations))
            == self.num_vehicles
        )
        self.cplex.add_constraint(
            self.cplex.sum(self.x[i, 0] for i in range(1, self.num_locations))
            == self.num_vehicles
        )

    def create_subtour_constraints(self):
        """
        Create the constraints that eliminate subtours (MTV).
        """

        for i in range(1, self.num_locations):
            for j in range(1, self.num_locations):
                if i == j:
                    continue

                self.cplex.add_constraint(
                    self.u[i - 1] - self.u[j - 1] + self.capacity * self.x[i, j]
                    <= self.capacity - self.get_location_demand(j)
                )

            self.cplex.add_constraint(self.u[i - 1] >= self.get_location_demand(i))

    def simplify_problem(self, qp: QuadraticProgram) -> QuadraticProgram:
        """
        Simplify the problem by removing unnecessary variables.
        """

        for i in range(len(self.distance_matrix)):
            qp = qp.substitute_variables({f"x_{i}_{i}": 0})

        return qp

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
        Get the name of a variable.
        """
        return f"x_{i}_{j}"

    def get_u_upper_bound(self) -> int:
        """
        Get the upper bound for the u variable.
        """

        return self.capacity
