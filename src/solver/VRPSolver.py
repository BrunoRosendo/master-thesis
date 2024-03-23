from abc import ABC, abstractmethod

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution


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
        use_deliveries (bool): Whether the problem uses deliveries or not.
        use_rpp (bool): Whether the problem uses the Ride Pooling Problem (RPP) or not.
    """

    def __init__(
        self,
        num_vehicles: int,
        capacities: int | list[int] | None,
        locations: list[tuple[int, int]],
        trips: list[tuple[int, int, int]],
        use_deliveries: bool,
        use_rpp: bool,
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
        self.use_deliveries = use_deliveries
        self.use_rpp = use_rpp
        self.distance_matrix = self.compute_distance()
        self.model = self.get_model()

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

    @abstractmethod
    def get_model(self) -> VRP:
        """
        Get the CVRPModel instance.
        """
        pass

    def solve(self) -> VRPSolution:
        """
        Solve the CVRP.
        """
        result = self._solve_cvrp()
        return self._convert_solution(result)
