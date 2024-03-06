class CVRPSolution:

    def __init__(
        self,
        num_vehicles,
        locations,
        objective,
        total_distance,
        routes,
        distances,
        loads=None,
    ):
        self.num_vehicles = num_vehicles
        self.locations = locations
        self.objective = objective
        self.total_distance = total_distance
        self.routes = routes
        self.distances = distances
        self.loads = loads
        self.use_capacity = loads is not None

    def display(self):
        """Display the solution."""
        print(f"Objective: {self.objective}\n")

        for vehicle_id in range(self.num_vehicles):
            print(f"Route for vehicle {vehicle_id}:")
            print("  ", end="")

            for i, node in enumerate(self.routes[vehicle_id]):
                if self.use_capacity:
                    print(f"{node} [{self.loads[vehicle_id][i]}P]", end="")
                else:
                    print(f"{node}", end="")
                if i < len(self.routes[vehicle_id]) - 1:
                    print(" -> ", end="")

            print(f"\nDistance of the route: {self.distances[vehicle_id]}m\n")

        print(f"Total distance of all routes: {self.total_distance}m")
