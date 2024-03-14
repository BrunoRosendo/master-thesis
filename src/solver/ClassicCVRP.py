from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from src.model.CVRPModel import CVRPModel
from src.model.CVRPSolution import CVRPSolution
from src.solver.CVRP import CVRP


class ClassicCVRP(CVRP):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with classic algorithms, using Google's OR Tools.
    """

    SOLUTION_STRATEGY = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    LOCAL_SEARCH_METAHEURISTIC = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
    DISTANCE_GLOBAL_SPAN_COST_COEFFICIENT = 100

    def _solve_cvrp(self) -> any:
        """
        Solve the CVRP using Google's OR Tools.
        """

        self.manager = pywrapcp.RoutingIndexManager(
            len(self.distance_matrix), self.num_vehicles, self.depot
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

        self.set_distance_dimension()
        if self.use_capacity:
            self.set_capacity_dimension()
        self.set_pickup_and_deliveries()

        search_parameters = self.get_search_parameters()
        or_solution = self.routing.SolveWithParameters(search_parameters)
        return or_solution

    def distance_callback(self, from_index: any, to_index: any) -> int:
        """Returns the distance between the two nodes."""

        # Convert from index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.distance_matrix[from_node][to_node]

    def demand_callback(self, from_index: any) -> int:
        """Returns the demand of the node."""

        # Convert from index to demands NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        return self.model.get_location_demand(from_node)

    def set_distance_dimension(self):
        """Set the distance dimension and cost for the problem."""

        # Define the cost of each arc.
        transit_callback_index = self.routing.RegisterTransitCallback(
            self.distance_callback
        )
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = "Distance"
        self.routing.AddDimension(
            transit_callback_index,
            0,
            3000,
            True,
            dimension_name,
        )
        self.distance_dimension = self.routing.GetDimensionOrDie(dimension_name)

        # Balances the distance between each vehicle.
        self.distance_dimension.SetGlobalSpanCostCoefficient(
            self.DISTANCE_GLOBAL_SPAN_COST_COEFFICIENT
        )

    def set_capacity_dimension(self):
        """Set the capacity dimension and constraint for the problem."""

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(
            self.demand_callback
        )

        capacities = (
            [self.capacities] * self.num_vehicles
            if self.same_capacity
            else self.capacities
        )

        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack - no extra capacity
            capacities,
            True,  # start counting at 0
            "Capacity",
        )

    def set_pickup_and_deliveries(self):
        """Set the pickup and delivery constraints for the problem."""

        for trip in self.trips:
            pickup_index = self.manager.NodeToIndex(trip[0])
            delivery_index = self.manager.NodeToIndex(trip[1])

            self.routing.AddPickupAndDelivery(pickup_index, delivery_index)

            self.routing.solver().Add(
                self.routing.VehicleVar(pickup_index)
                == self.routing.VehicleVar(delivery_index)
            )

            self.routing.solver().Add(
                self.distance_dimension.CumulVar(pickup_index)
                <= self.distance_dimension.CumulVar(delivery_index)
            )

    def get_search_parameters(self) -> pywrapcp.DefaultRoutingSearchParameters:
        """Returns the search parameters for the problem."""

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = self.SOLUTION_STRATEGY
        search_parameters.local_search_metaheuristic = self.LOCAL_SEARCH_METAHEURISTIC

        return search_parameters

    def _convert_solution(self, result: any) -> CVRPSolution:
        """Converts OR-Tools result to CVRP solution."""

        routes = []
        loads = []
        distances = []
        total_distance = 0

        for vehicle_id in range(self.num_vehicles):
            index = self.routing.Start(vehicle_id)

            route = []
            route_loads = []
            route_distance = 0
            cur_load = 0

            while not self.routing.IsEnd(index):
                cur_load += self.demand_callback(
                    index
                )  # This assumes each location only has one vehicle visiting it

                previous_index = index
                index = result.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
                route.append(self.manager.IndexToNode(previous_index))
                route_loads.append(cur_load)

            route.append(self.manager.IndexToNode(index))
            route_loads.append(cur_load)

            routes.append(route)
            distances.append(route_distance)
            loads.append(route_loads)
            total_distance += route_distance

        return CVRPSolution(
            self.num_vehicles,
            self.locations,
            result.ObjectiveValue(),
            total_distance,
            routes,
            distances,
            self.depot,
            loads if self.use_capacity else None,
        )

    def get_model(self) -> CVRPModel:
        """
        Get the CVRPModel instance.
        """

        return CVRPModel(
            self.num_vehicles, self.trips, self.depot, self.distance_matrix
        )
