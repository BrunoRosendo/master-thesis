from abc import ABC, abstractmethod


class CVRP(ABC):
    """
    Abstract class for solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        vehicle_capacities (list): List of vehicle capacities.
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
        locations (list): List of coordinates for each location.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
        num_vehicles (int): Number of vehicles available.
    """

    def __init__(self, vehicle_capacities, depot, locations, trips):
        self.vehicle_capacities = vehicle_capacities
        self.depot = depot
        self.locations = locations
        self.trips = trips
        self.num_vehicles = len(vehicle_capacities)
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

    @abstractmethod
    def solve(self):
        """
        Solve the CVRP.
        """
        pass
