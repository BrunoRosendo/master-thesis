from qiskit_optimization import QuadraticProgram

from src.model.CPLEXModel import CPLEXModel


class DiffCapModel(CPLEXModel):
    """
    A class to represent a CPLEX math formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        capacities (list): List of vehicle capacities.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        cplex (Model): CPLEX model for the CVRP
    """

    def __init__(self, num_vehicles, trips, depot, distance_matrix, capacities):
        self.capacities = capacities
        super().__init__(num_vehicles, trips, depot, distance_matrix)

    def create_vars(self):
        """
        Create the variables for the CPLEX model.
        """

        self.x = self.cplex.binary_var_cube(
            self.num_locations, self.num_locations, self.num_vehicles, name="x"
        )

        self.u = self.cplex.integer_var_list(range(1, self.num_locations), name="u")

    def create_objective(self):
        """
        Create the objective function for the CPLEX model.
        """

        objective = self.cplex.sum(
            self.distance_matrix[i][j] * self.x[i, j, k]
            for i in range(self.num_locations)
            for j in range(self.num_locations)
            for k in range(self.num_vehicles)
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
                    self.x[i, j, k]
                    for j in range(self.num_locations)
                    for k in range(self.num_vehicles)
                    if i != j
                )
                == 1
            )
            self.cplex.add_constraint(
                self.cplex.sum(
                    self.x[j, i, k]
                    for j in range(self.num_locations)
                    for k in range(self.num_vehicles)
                    if i != j
                )
                == 1
            )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle starts and ends at the depot.
        """

        for k in range(self.num_vehicles):
            self.cplex.add_constraint(
                self.cplex.sum(self.x[0, i, k] for i in range(1, self.num_locations))
                == 1
            )
            self.cplex.add_constraint(
                self.cplex.sum(self.x[i, 0, k] for i in range(1, self.num_locations))
                == 1
            )

    def create_subtour_constraints(self):
        """
        Create the constraints that eliminate subtours (MTV).
        """

        for i in range(1, self.num_locations):
            for k in range(self.num_vehicles):
                self.cplex.add_constraint(
                    self.u[i - 1] <= self.capacities[k]
                )  # TODO: not correct, this will use the smallest capacity

                for j in range(1, self.num_locations):
                    if i == j:
                        continue

                    self.cplex.add_constraint(
                        self.u[i - 1]
                        - self.u[j - 1]
                        + self.capacities[k] * self.x[i, j, k]
                        <= self.capacities[k] - self.get_location_demand(j)
                    )

            self.cplex.add_constraint(self.u[i - 1] >= self.get_location_demand(i))

    def simplify(self, qp: QuadraticProgram) -> QuadraticProgram:  # TODO
        """
        Simplify the problem by removing unnecessary variables.
        """

        return qp
