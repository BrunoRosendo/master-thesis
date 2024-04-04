from src.model.cplex.CplexVRP import CplexVRP


class CplexConstantCVRP(CplexVRP):
    """
    A class to represent a CPLEX math formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        capacity (int): Capacity of each vehicle.
        locations (list): List of coordinates for each location.
        cplex (Model): CPLEX model for the CVRP
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
    """

    def __init__(
        self,
        num_vehicles: int,
        distance_matrix: list[list[int]],
        capacity: int | None,
        locations: list[tuple[int, int]],
        simplify: bool,
    ):
        self.capacity = capacity
        super().__init__(num_vehicles, [], distance_matrix, locations, False, simplify)

    def create_vars(self):
        """
        Create the variables for the CPLEX model.
        """

        self.x = self.cplex.binary_var_matrix(
            self.num_locations, self.num_locations, name="x"
        )

        self.u = self.cplex.integer_var_list(
            range(1, self.num_locations), name="u", ub=self.get_u_upper_bound()
        )
        for i, var in enumerate(self.u):
            var.set_lb(self.get_u_lower_bound(i + 1))

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

        a = self.cplex.add_constraint(
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
