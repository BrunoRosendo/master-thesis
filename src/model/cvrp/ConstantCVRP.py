from typing import Callable

from src.model.VRPSolution import DistanceUnit
from src.model.cvrp.InfiniteCVRP import InfiniteCVRP
from src.solver.cost_functions import manhattan_distance


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
        simplify: bool = True,
        cost_function: Callable[
            [list[tuple[float, float]], DistanceUnit], list[list[float]]
        ] = manhattan_distance,
        depot: int | None = 0,
        distance_matrix: list[list[float]] | None = None,
        location_names: list[str] | None = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
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
