class CVRPModel:
    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        depot: int,
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        use_deliveries: bool,
    ):
        """
        A class to represent a formulation of the CVRP model.

        Attributes:
            num_vehicles (int): Number of vehicles available.
            trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of customers for a trip.
            depot (int): Index of the depot, which is the starting and ending point for each vehicle.
            distance_matrix (list): Matrix with the distance between each pair of locations.
            locations (list): List of coordinates for each location.
            use_deliveries (bool): Whether the problem uses deliveries or not.
        """

        self.num_vehicles = num_vehicles
        self.trips = trips
        self.depot = depot
        self.distance_matrix = distance_matrix
        self.num_locations = len(distance_matrix)
        self.locations = locations
        self.use_deliveries = use_deliveries

    def get_location_demand(self, idx: int) -> int:
        """
        Get the demand for a location.
        """
        pickup_demand = sum(trip[2] for trip in self.trips if idx == trip[0])
        if not self.use_deliveries:
            return pickup_demand

        delivery_demand = sum(trip[2] for trip in self.trips if idx == trip[1])
        return pickup_demand - delivery_demand
