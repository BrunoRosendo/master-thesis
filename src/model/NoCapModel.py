from src.model.SameCapModel import SameCapModel


class NoCapModel(SameCapModel):
    """
    A class to represent a CPLEX math formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        cplex (Model): CPLEX model for the CVRP
    """

    def __init__(self, num_vehicles, trips, depot, distance_matrix):
        super().__init__(num_vehicles, trips, depot, distance_matrix, None)

    def create_subtour_constraints(self):
        """
        Create the constraints that eliminate subtours (MTV).
        """

        for i in range(1, self.num_locations):
            for j in range(1, self.num_locations):
                if i == j:
                    continue

                self.cplex.add_constraint(
                    self.u[i - 1] - self.u[j - 1] + self.num_locations * self.x[i, j]
                    <= self.num_locations - 1
                )

            self.cplex.add_constraint(self.u[i - 1] >= 1)
            self.cplex.add_constraint(self.u[i - 1] <= self.num_locations)
