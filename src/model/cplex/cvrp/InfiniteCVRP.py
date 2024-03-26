from src.model.cplex.CplexVRP import CplexVRP


class InfiniteCVRP(CplexVRP):
    """
    A class to represent a CPLEX math formulation of the CVRP model with all vehicles having the same capacity.

    This model should always be simplified, since some constraints assume the simplification.

    Attributes:
        num_vehicles (int): Number of vehicles available.
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
        locations: list[tuple[int, int]],
        use_deliveries: bool,
        simplify: bool,
    ):
        self.num_steps = len(distance_matrix) + 1
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

        self.x = self.cplex.binary_var_cube(
            self.num_vehicles, self.num_locations, self.num_steps, name="x"
        )

    def create_objective(self):
        """
        Create the objective function for the CPLEX model.
        """

        objective = self.cplex.sum(
            self.distance_matrix[i][j]
            / self.normalization_factor
            * self.x[k, i, s]
            * self.x[k, j, s + 1]
            for k in range(self.num_vehicles)
            for i in range(self.num_locations)
            for j in range(self.num_locations)
            for s in range(self.num_steps - 1)
        )

        self.cplex.minimize(objective)

    def create_constraints(self):
        """
        Create the constraints for the CPLEX model.
        """

        self.create_location_constraints()
        self.create_vehicle_constraints()

    def create_location_constraints(self):
        """
        Create the constraints that ensure each location is visited exactly once.
        """

        for i in range(1, self.num_locations):
            self.cplex.add_constraint(
                self.cplex.sum(
                    self.x[k, i, s]
                    for k in range(self.num_vehicles)
                    for s in range(self.num_steps)
                )
                == 1
            )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle only visits one location at a time.
        """

        for k in range(self.num_vehicles):
            for s in range(self.num_steps):
                self.cplex.add_constraint(
                    self.cplex.sum(self.x[k, i, s] for i in range(self.num_locations))
                    == 1
                )

    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that are relevant for the solution.
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
