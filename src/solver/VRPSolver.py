import time
from abc import ABC, abstractmethod
from typing import Callable, Any

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution, DistanceUnit
from src.model.qubo.QuboVRP import QuboVRP


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
        use_rpp (bool): Whether the problem uses the Ride Pooling Problem (RPP) or not.
        track_progress (bool): Whether to track the progress of the solver or not.
        simplify (bool): Whether to simplify the problem by removing unnecessary variables.
        distance_function (Callable): Function to compute the distance between two locations.
        model (QuboVRP): VRP instance of the model.
        run_time (int): Time taken to run the solver (measured locally).
        location_names (list): List of names for each location. Optional.
        distance_unit (DistanceUnit): Unit of distance used in the problem.
    """

    def __init__(
        self,
        num_vehicles: int,
        capacities: int | list[int] | None,
        locations: list[tuple[int, int]],
        trips: list[tuple[int, int, int]],
        use_rpp: bool,
        track_progress: bool,
        distance_function: Callable[[tuple[int, int], tuple[int, int]], float],
        simplify: bool = True,
        distance_matrix: list[list[float]] = None,
        location_names: list[str] = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
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
        self.distance_function = distance_function
        self.location_names = location_names
        self.distance_unit = distance_unit
        self.run_time: int | None = None

        if distance_matrix is None:
            self.distance_matrix = self.compute_distance()
        else:
            self.distance_matrix = distance_matrix

        self.model = self.get_model()

    def compute_distance(self) -> list[list[float]]:
        """
        Compute the distance matrix between each pair of locations using Manhattan.
        """
        return [
            [
                self.distance_function(from_location, to_location)
                for to_location in self.locations
            ]
            for from_location in self.locations
        ]

    @abstractmethod
    def _solve_cvrp(self) -> Any:
        """
        Solve the CVRP with a specific solver.
        """
        pass

    @abstractmethod
    def _convert_solution(self, result: Any, local_run_time: float) -> VRPSolution:
        """
        Convert the result from the solver to a CVRP solution.
        """
        pass

    def solve(self) -> VRPSolution:
        """
        Solve the CVRP.
        """

        result, execution_time = self.measure_time(self._solve_cvrp)
        return self._convert_solution(result, execution_time)

    @abstractmethod
    def get_model(self) -> VRP:
        """
        Get a VRP instance of the model.
        """
        pass

    @staticmethod
    def measure_time(
        fun: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> tuple[Any, int]:
        """
        Measure the execution time of a function.
        Returns the result and the execution time in microseconds.
        """

        start_time = time.perf_counter_ns()
        result = fun(*args, **kwargs)
        execution_time = (
            time.perf_counter_ns() - start_time
        ) // 1000  # Convert to microseconds

        return result, execution_time
