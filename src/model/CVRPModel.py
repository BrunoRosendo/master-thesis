class CVRPModel:
    def __init__(self, num_vehicles, trips, depot, distance_matrix):
        """
        A class to represent a formulation of the CVRP model.

        Attributes:
            num_vehicles (int): Number of vehicles available.
            trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
            depot (int): Index of the depot, which is the starting and ending point for each vehicle.
            distance_matrix (list): Matrix with the distance between each pair of locations.
        """

        self.num_vehicles = num_vehicles
        self.trips = trips
        self.depot = depot
        self.distance_matrix = distance_matrix
        self.num_locations = len(distance_matrix)

    def get_location_demand(self, idx: int) -> int:
        """
        Get the demand for a location.
        """
        return 3  # TODO: Change once pickup and delivery is implemented in qubo
        # pickup_demand = sum(trip[2] for trip in self.trips if idx == trip[0])
        # delivery_demand = sum(trip[2] for trip in self.trips if idx == trip[1])
        # return pickup_demand - delivery_demand
