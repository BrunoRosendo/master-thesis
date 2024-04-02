from src.model.dwave.cvrp.DWaveConstantCVRP import DWaveConstantCVRP


class DWaveInfiniteCVRP(DWaveConstantCVRP):
    """
    A class to represent a DWave Ocean formulation of the CVRP model with all vehicles having infinite capacity.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        locations (list): List of coordinates for each location.
        use_deliveries (bool): Whether the problem uses deliveries or not.
        cqm (ConstrainedQuadraticModel): DWave Ocean model for the CVRP
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
    """

    def __init__(
        self,
        num_vehicles: int,
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        simplify: bool,
    ):
        super().__init__(num_vehicles, distance_matrix, locations, simplify, None)

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
                        + self.num_locations * self.x_var(i, j)
                        <= self.num_locations - 1
                    )

    def get_u_lower_bound(self, i: int) -> int:
        """
        Get the lower bound for the u variable, at the given index.
        """

        return 1

    def get_u_upper_bound(self):
        """
        Get the upper bound for the variable u.
        """

        return self.num_locations
