from abc import ABC, abstractmethod


class CVRP(ABC):
    """
    Abstract class for solving the Capacitated Vehicle Routing Problem (CVRP).

    Attributes:
        vehicle_capacities (list): List of vehicle capacities.
        depot (int): Index of the depot, which is the starting and ending point for each vehicle.
        locations (list): List of coordinates for each location.
        demands (list): List of demands for each location, corresponding to the quantity of customers to be picked up.
        num_vehicles (int): Number of vehicles available.
    """

    def __init__(self, vehicle_capacities, depot, locations, demands):
        self.vehicle_capacities = vehicle_capacities
        self.depot = depot
        self.locations = locations
        self.demands = demands
        self.num_vehicles = len(vehicle_capacities)

    @abstractmethod
    def solve(self):
        """
        Solve the CVRP.
        """
        pass
