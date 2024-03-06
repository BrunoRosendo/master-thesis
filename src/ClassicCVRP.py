from CVRP import CVRP
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


class ClassicCVRP(CVRP):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with classic algorithms, using Google's OR Tools.
    """

    def distance_callback(self, from_index, to_index):
        """Returns the distance between the two nodes."""

        # Convert from index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.distance_matrix[from_node][to_node]

    def demand_callback(self, from_index):
        """Returns the demand of the node."""

        # Convert from index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)

        pickup_demand = sum(trip[2] for trip in self.trips if from_node == trip[0])
        delivery_demand = sum(trip[2] for trip in self.trips if from_node == trip[1])

        return pickup_demand - delivery_demand

    def solve(self):
        """
        Solve the CVRP using Google's OR Tools.
        """

        self.manager = pywrapcp.RoutingIndexManager(
            len(self.distance_matrix), self.num_vehicles, self.depot
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

        # Define the cost of each arc.
        transit_callback_index = self.routing.RegisterTransitCallback(
            self.distance_callback
        )
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(
            self.demand_callback
        )
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack - no extra capacity
            self.vehicle_capacities,  # vehicle maximum capacities
            True,  # start counting at 0
            "Capacity",
        )

        # Add Distance dimension.
        distance_dimension = "Distance"
        self.routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            3000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            distance_dimension,
        )
        distance_dimension = self.routing.GetDimensionOrDie(distance_dimension)

        # Balances the distance between each vehicle.
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Define Transportation Requests.
        for trip in self.trips:
            pickup_index = self.manager.NodeToIndex(trip[0])
            delivery_index = self.manager.NodeToIndex(trip[1])
            self.routing.AddPickupAndDelivery(pickup_index, delivery_index)
            self.routing.solver().Add(
                self.routing.VehicleVar(pickup_index)
                == self.routing.VehicleVar(delivery_index)
            )
            self.routing.solver().Add(
                distance_dimension.CumulVar(pickup_index)
                <= distance_dimension.CumulVar(delivery_index)
            )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )

        # Solve the problem.
        solution = self.routing.SolveWithParameters(search_parameters)
        if solution:
            self.print_solution(solution)

    def print_solution(self, solution):
        """Prints solution on console."""

        print(f"Objective: {solution.ObjectiveValue()}")

        total_distance = 0
        for vehicle_id in range(self.num_vehicles):
            index = self.routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_distance = 0
            route_load = 0

            while not self.routing.IsEnd(index):
                route_load += self.demand_callback(
                    index
                )  # This assumes each location only has one vehicle visiting it
                plan_output += (
                    f" {self.manager.IndexToNode(index)} Load({route_load}) -> "
                )
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            plan_output += f"{self.manager.IndexToNode(index)} Load({route_load})\n"
            plan_output += f"Distance of the route: {route_distance}m\n"
            print(plan_output)
            total_distance += route_distance

        print(f"Total Distance of all routes: {total_distance}m")
