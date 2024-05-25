from src.model.VRPSolution import DistanceUnit


class VRP:
    """
    A class to represent a formulation of the VRP model.

    Attributes:
        num_vehicles (int): Number of vehicles available.
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of
            customers for a trip.
        distance_matrix (list): Matrix with the distance between each pair of locations.
        locations (list): List of coordinates for each location.
        use_deliveries (bool): Whether the problem uses deliveries or not.
        depot (int | None): Index of the depot location, if it exists.
        location_names (list): List of names for each location. Optional.
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[float]],
        locations: list[tuple[float, float]],
        use_deliveries: bool,
        depot: int | None = 0,
        location_names: list[str] = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
    ):

        self.num_vehicles = num_vehicles
        self.trips = trips
        self.distance_matrix = distance_matrix
        self.num_locations = len(distance_matrix)
        self.locations = locations
        self.use_deliveries = use_deliveries
        self.depot = depot
        self.location_names = location_names
        self.distance_unit = distance_unit

    def get_location_demand(self, idx: int) -> int:
        """
        Get the demand for a location.
        """
        pickup_demand = self.get_location_pickup(idx)
        if not self.use_deliveries:
            return max(pickup_demand, 1)

        delivery_demand = self.get_location_delivery(idx)
        return pickup_demand - delivery_demand

    def get_location_pickup(self, idx: int) -> int:
        """
        Get the demand for a location.
        """
        pickup_demand = sum(trip[2] for trip in self.trips if idx == trip[0])
        return pickup_demand

    def get_location_delivery(self, idx: int) -> int:
        """
        Get the demand for a location.
        """
        delivery_demand = sum(trip[2] for trip in self.trips if idx == trip[1])
        return delivery_demand
