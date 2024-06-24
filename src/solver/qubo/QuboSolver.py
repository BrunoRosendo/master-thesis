from abc import ABC

from src.model.qubo.QuboVRP import QuboVRP
from src.model.qubo.cvrp.ConstantCVRP import ConstantCVRP
from src.model.qubo.cvrp.InfiniteCVRP import InfiniteCVRP
from src.model.qubo.cvrp.MultiCVRP import MultiCVRP
from src.model.qubo.rpp.CapacityRPP import CapacityRPP
from src.model.qubo.rpp.InfiniteRPP import InfiniteRPP
from src.solver.VRPSolver import VRPSolver


class QuboSolver(VRPSolver, ABC):
    """
    Abstract class for solving the Capacitated Vehicle Routing Problem (CVRP) with QUBO algorithm.
    """

    def get_model(self) -> QuboVRP:
        if self.use_rpp:
            if self.use_capacity:
                return CapacityRPP(
                    self.num_vehicles,
                    self.trips,
                    self.distance_matrix,
                    self.locations,
                    (
                        [self.capacities] * self.num_vehicles
                        if self.same_capacity
                        else self.capacities
                    ),
                    self.location_names,
                    self.distance_unit,
                )
            return InfiniteRPP(
                self.num_vehicles,
                self.trips,
                self.distance_matrix,
                self.locations,
                self.location_names,
                self.distance_unit,
            )

        if not self.use_capacity:
            return InfiniteCVRP(
                self.num_vehicles,
                self.distance_matrix,
                self.locations,
                self.simplify,
                self.location_names,
                self.distance_unit,
            )
        if self.same_capacity:
            return ConstantCVRP(
                self.num_vehicles,
                self.trips,
                self.distance_matrix,
                self.capacities,
                self.locations,
                self.simplify,
                self.location_names,
                self.distance_unit,
            )

        return MultiCVRP(
            self.num_vehicles,
            self.trips,
            self.distance_matrix,
            self.capacities,
            self.locations,
            self.location_names,
            self.distance_unit,
        )
