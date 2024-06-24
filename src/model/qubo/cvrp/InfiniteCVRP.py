from src.model.VRPSolution import DistanceUnit
from src.model.qubo.cvrp.ConstantCVRP import ConstantCVRP


class InfiniteCVRP(ConstantCVRP):
    """
    A class to represent a QUBO math formulation of the CVRP model with all vehicles having infinite capacity.
    """

    def __init__(
        self,
        num_vehicles: int,
        distance_matrix: list[list[float]],
        locations: list[tuple[float, float]],
        simplify: bool,
        location_names: list[str] = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
    ):
        super().__init__(
            num_vehicles,
            [],
            distance_matrix,
            None,
            locations,
            simplify,
            location_names,
            distance_unit,
        )

    def create_subtour_constraints(self):
        """
        Create the constraints that eliminate subtours (MTV).
        """

        self.constraints.extend(
            self.u[i - 1] - self.u[j - 1] + self.num_locations * self.x_var(i, j)
            <= self.num_locations - 1
            for i in range(1, self.num_locations)
            for j in range(1, self.num_locations)
            if i != j
        )

    def get_u_lower_bound(self, i: int) -> int:
        """
        Get the lower bound for the auxiliary variable, at the given index.
        """

        return 1

    def get_u_upper_bound(self):
        """
        Get the upper bound for the variable u.
        """

        return self.num_locations

    def get_capacity(self) -> int | list[int] | None:
        """
        Get the capacity of the vehicles. In this case, it is infinite.
        """

        return None
