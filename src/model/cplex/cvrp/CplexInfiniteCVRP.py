from src.model.cplex.cvrp.CplexConstantCVRP import CplexConstantCVRP


class CplexInfiniteCVRP(CplexConstantCVRP):
    """
    A class to represent a CPLEX math formulation of the CVRP model with all vehicles having infinite capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        locations (list): List of coordinates for each location.
        cplex (Model): CPLEX model for the CVRP
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
    """

    def __init__(
        self,
        num_vehicles: int,
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        simplify: bool,
    ):
        super().__init__(num_vehicles, distance_matrix, None, locations, simplify)

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

    def get_u_upper_bound(self):
        """
        Get the upper bound for the variable u.
        """

        return self.num_locations
