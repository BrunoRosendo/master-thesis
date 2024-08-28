from typing import Callable

from src.model.VRP import DistanceUnit
from src.model.cvrp.InfiniteCVRP import InfiniteCVRP


class ConstantCVRP(InfiniteCVRP):
    """
    A class to represent a QUBO math formulation of the CVRP model with all vehicles having the same capacity.

    Attributes:
        capacity (int): Capacity of each vehicle.
    """

    def __init__(
        self,
        num_vehicles: int,
        locations: list[tuple[float, float]],
        demands: list[int],
        capacity: int | None,
        simplify: bool,
        cost_function: Callable[
            [list[tuple[float, float]], DistanceUnit], list[list[float]]
        ],
        depot: int | None,
        distance_matrix: list[list[float]] | None,
        location_names: list[str] | None,
        distance_unit: DistanceUnit,
    ):
        self.capacity = capacity
        super().__init__(
            num_vehicles,
            locations,
            demands,
            simplify,
            cost_function,
            depot,
            distance_matrix,
            location_names,
            distance_unit,
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
