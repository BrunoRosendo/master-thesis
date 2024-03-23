from src.model.cplex.rpp.InfiniteRPP import InfiniteRPP


class CapacityRPP(InfiniteRPP):
    """
    A class to represent a CPLEX math formulation of the RPP model with a capacity constraint for each vehicle.
    Note that this model assumes a starting point with no cost to the first node for each vehicle,
    as to avoid the need for a depot.

    This model should always be simplified, since some constraints assume the simplification.

    Attributes:
        capacities (list): Capacities for each vehicle.
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        capacities: list[int],
        simplify: bool,
    ):
        self.capacities = capacities
        super().__init__(num_vehicles, trips, distance_matrix, locations, simplify)

    def create_constraints(self):
        """
        Create the constraints for the CPLEX model.
        """

        super().create_constraints()
        self.create_capacity_constraints()

    def create_capacity_constraints(self):
        """
        Create the capacity constraints for the CPLEX model.
        """

        for k in range(self.num_vehicles):
            for cur_step in range(1, self.num_steps):  # start has no demand
                self.cplex.add_constraint(
                    self.cplex.sum(
                        self.get_location_demand(self.used_locations_indices[i - 1])
                        * self.x[k, i, s]
                        for i in range(1, self.num_used_locations + 1)
                        for s in range(cur_step + 1)
                    )
                    <= self.capacities[k]
                )
