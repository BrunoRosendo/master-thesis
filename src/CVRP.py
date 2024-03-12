from abc import ABC, abstractmethod


class CVRP(ABC):
    """
    Abstract class for solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        vehicle_capacities (int | list): List of vehicle capacities, if available.
        use_capacity (bool): Whether the problem uses vehicle capacities or not
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
        locations (list): List of coordinates for each location.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        num_vehicles (int): Number of vehicles available.
        distance_matrix (list): Matrix with the distance between each pair of locations.
    """

    def __init__(self, vehicles, depot, locations, trips):
        if type(vehicles) == int:
            self.num_vehicles = vehicles
            self.use_capacity = False
        else:
            self.num_vehicles = len(vehicles)
            self.vehicle_capacities = vehicles
            self.use_capacity = True

        self.depot = depot
        self.locations = locations
        self.trips = trips
        self.distance_matrix = self.compute_distance()

    def compute_distance(self):
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

    def get_location_demand(self, idx):
        """
        Get the demand for a location.
        """
        return 1  # TODO: Change once pickup and delivery is implemented in qubo
        # pickup_demand = sum(trip[2] for trip in self.trips if idx == trip[0])
        # delivery_demand = sum(trip[2] for trip in self.trips if idx == trip[1])
        # return pickup_demand - delivery_demand

    @abstractmethod
    def _solve_cvrp(self):
        """
        Solve the CVRP with a specific solver.
        """
        pass

    @abstractmethod
    def _convert_solution(self, result):
        """
        Convert the result from the solver to a CVRP solution.
        """
        pass

    def solve(self):
        """
        Solve the CVRP.
        """
        result = self._solve_cvrp()
        return self._convert_solution(result)
