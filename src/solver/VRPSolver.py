from abc import ABC, abstractmethod

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution
from src.model.qubo.QuboVRP import QuboVRP
from src.model.qubo.cvrp.ConstantCVRP import ConstantCVRP
from src.model.qubo.cvrp.InfiniteCVRP import InfiniteCVRP
from src.model.qubo.cvrp.MultiCVRP import MultiCVRP
from src.model.qubo.rpp.CapacityRPP import CapacityRPP
from src.model.qubo.rpp.InfiniteRPP import InfiniteRPP


class VRPSolver(ABC):
    """
    Abstract class for solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        num_vehicles (int): Number of vehicles available.
        capacities (int | list | None): Vehicle capacity or list of vehicle capacities, if used.
        use_capacity (bool): Whether the problem uses vehicle capacities or not
        same_capacity (bool): Whether all vehicles have the same capacity or not
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
        locations (list): List of coordinates for each location.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        model (VRP): The CVRP model instance.
        use_rpp (bool): Whether the problem uses the Ride Pooling Problem (RPP) or not.
        track_progress (bool): Whether to track the progress of the solver or not.
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
    """

    def __init__(
        self,
        num_vehicles: int,
        capacities: int | list[int] | None,
        locations: list[tuple[int, int]],
        trips: list[tuple[int, int, int]],
        use_rpp: bool,
        track_progress: bool,
        simplify: bool = True,
    ):
        if capacities is None:
            self.use_capacity = False
            self.same_capacity = True  # Infinitely large capacity
        elif isinstance(capacities, int):
            self.use_capacity = True
            self.same_capacity = True
        else:
            self.use_capacity = True
            self.same_capacity = False

        self.depot = 0
        self.num_vehicles = num_vehicles
        self.capacities = capacities
        self.locations = locations
        self.trips = trips
        self.use_rpp = use_rpp
        self.track_progress = track_progress
        self.simplify = simplify
        self.distance_matrix = self.compute_distance()
        self.qubo = self.get_qubo()

    def compute_distance(self) -> list[list[int]]:
        """
        Compute the distance matrix between each pair of locations using Manhattan.
        """

        distance_matrix = []
        for from_location in self.locations:
            row = []
            for to_location in self.locations:
                row.append(
                    abs(from_location[0] - to_location[0])
                    + abs(from_location[1] - to_location[1])
                )
            distance_matrix.append(row)
        return distance_matrix

    @abstractmethod
    def _solve_cvrp(self) -> any:
        """
        Solve the CVRP with a specific solver.
        """
        pass

    @abstractmethod
    def _convert_solution(self, result: any) -> VRPSolution:
        """
        Convert the result from the solver to a CVRP solution.
        """
        pass

    def solve(self) -> VRPSolution:
        """
        Solve the CVRP.
        """
        result = self._solve_cvrp()
        return self._convert_solution(result)

    def get_qubo(self) -> QuboVRP:
        """
        Get a qubo instance of the CVRPModel.
        """

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
                )
            return InfiniteRPP(
                self.num_vehicles,
                self.trips,
                self.distance_matrix,
                self.locations,
            )

        if not self.use_capacity:
            return InfiniteCVRP(
                self.num_vehicles, self.distance_matrix, self.locations, self.simplify
            )
        if self.same_capacity:
            return ConstantCVRP(
                self.num_vehicles,
                self.distance_matrix,
                self.capacities,
                self.locations,
                self.simplify,
            )

        return MultiCVRP(
            self.num_vehicles, self.distance_matrix, self.capacities, self.locations
        )
