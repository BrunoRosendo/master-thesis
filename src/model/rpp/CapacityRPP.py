from typing import Callable

from src.model.VRP import DistanceUnit
from src.model.rpp.InfiniteRPP import InfiniteRPP
from src.solver.cost_functions import manhattan_distance


class CapacityRPP(InfiniteRPP):
    """
    A class to represent a QUBO math formulation of the RPP model with a capacity constraint for each vehicle.
    Note that this model assumes a starting point with no cost to the first node for each vehicle,
    as to avoid the need for a depot.

    This model is always be simplified, since some constraints assume the simplification.

    Attributes:
        capacities (list): Capacities for each vehicle.
    """

    def __init__(
        self,
        num_vehicles: int,
        locations: list[tuple[float, float]],
        trips: list[tuple[int, int, int]],
        capacities: list[int],
        cost_function: Callable[
            [list[tuple[float, float]], DistanceUnit], list[list[float]]
        ] = manhattan_distance,
        distance_matrix: list[list[float]] | None = None,
        location_names: list[str] | None = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
    ):
        self.capacities = capacities
        super().__init__(
            num_vehicles,
            locations,
            trips,
            cost_function,
            distance_matrix,
            location_names,
            distance_unit,
        )

    def create_constraints(self):
        """
        Create the constraints for the RPP model.
        """

        super().create_constraints()
        self.create_capacity_constraints()

    def create_capacity_constraints(self):
        """
        Create the capacity constraints for the RPP model.
        """

        self.constraints.extend(
            self.model.sum(
                self.get_location_demand(self.used_locations_indices[i - 1])
                * self.x_var(k, i, s)
                for i in range(1, self.num_used_locations)
                for s in range(cur_step + 1)
            )
            <= self.capacities[k]
            for k in range(self.num_vehicles)
            for cur_step in range(1, self.num_steps)
        )
