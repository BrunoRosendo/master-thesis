from src.model.cplex.cvrp.InfiniteCVRP import InfiniteCVRP


class CapacityCVRP(InfiniteCVRP):
    """
    A class to represent a CPLEX math formulation of the CVRP model with a capacity constraint for each vehicle.

    This model should always be simplified, since some constraints assume the simplification.

    Attributes:
        capacities (list): Capacities for each vehicle.
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        depot: int,
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        use_deliveries: bool,
        capacities: list[int],
        simplify: bool,
    ):
        self.capacities = capacities
        super().__init__(
            num_vehicles,
            trips,
            depot,
            distance_matrix,
            locations,
            use_deliveries,
            simplify,
        )

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
            for cur_step in range(1, self.num_steps):  # depot has no demand
                self.cplex.add_constraint(
                    self.cplex.sum(
                        self.get_location_demand(i) * self.x[k, i, s]
                        for i in range(1, self.num_locations)
                        for s in range(cur_step + 1)
                    )
                    <= self.capacities[k]
                )
