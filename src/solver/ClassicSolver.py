from typing import Any

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution
from src.model.rpp.InfiniteRPP import InfiniteRPP
from src.solver.VRPSolver import VRPSolver

DEFAULT_SOLUTION_STRATEGY = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
DEFAULT_LOCAL_SEARCH_METAHEURISTIC = (
    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
)
DEFAULT_DISTANCE_GLOBAL_SPAN_COST_COEFFICIENT = 1
DEFAULT_MAX_DISTANCE_CAPACITY = 90000
DEFAULT_TIME_LIMIT_SECONDS = 10


class ClassicSolver(VRPSolver):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with classic algorithms, using Google's OR Tools.
    Simplifying the model has no impact on this solver.

    Attributes:
    - solution_strategy (int): The strategy to use to find the first solution.
    - local_search_metaheuristic (int): The local search metaheuristic to use.
    - distance_global_span_cost_coefficient (int): The coefficient for the global span cost.
    - distance_dimension (pywrapcp.RoutingDimension): The distance dimension for the problem.
    - use_capacity (bool): Whether the problem uses vehicle capacities or not
    """

    def __init__(
        self,
        model: VRP,
        track_progress: bool = True,
        solution_strategy: int = DEFAULT_SOLUTION_STRATEGY,
        local_search_metaheuristic: int = DEFAULT_LOCAL_SEARCH_METAHEURISTIC,
        distance_global_span_cost_coefficient: int = DEFAULT_DISTANCE_GLOBAL_SPAN_COST_COEFFICIENT,
        time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
        max_distance_capacity: int = DEFAULT_MAX_DISTANCE_CAPACITY,
    ):
        self.use_rpp = isinstance(model, InfiniteRPP)

        if self.use_rpp:
            self.remove_unused_locations(model)

        super().__init__(model, track_progress)

        self.solution_strategy = solution_strategy
        self.local_search_metaheuristic = local_search_metaheuristic
        self.distance_global_span_cost_coefficient = (
            distance_global_span_cost_coefficient
        )
        self.time_limit_seconds = time_limit_seconds
        self.max_distance_capacity = max_distance_capacity

        if hasattr(model, "capacity") or hasattr(model, "capacities"):
            self.use_capacity = True
        else:
            self.use_capacity = False

        if self.use_rpp:
            self.add_dummy_depot()
        else:
            self.depot = model.depot

        self.distance_dimension = None

    def _solve_cvrp(self) -> Any:
        """
        Solve the CVRP using Google's OR Tools.
        """

        self.manager = pywrapcp.RoutingIndexManager(
            len(self.model.distance_matrix), self.model.num_vehicles, self.depot
        )
        self.routing = pywrapcp.RoutingModel(self.manager)

        self.set_distance_dimension()
        if self.use_capacity:
            self.set_capacity_dimension()
        if self.use_rpp:
            self.set_pickup_and_deliveries()

        search_parameters = self.get_search_parameters()
        or_solution, self.run_time = self.measure_time(
            self.routing.SolveWithParameters, search_parameters
        )

        if or_solution is None:
            raise Exception("The solution is infeasible, aborting!")

        return or_solution

    def distance_callback(self, from_index: Any, to_index: Any) -> float:
        """Returns the distance between the two nodes."""

        # Convert from index to distance matrix NodeIndex.
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.model.distance_matrix[from_node][to_node]

    def demand_callback(self, from_index: Any) -> int:
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

        # Distance dimension is used for pickup-delivery order.
        if self.use_rpp:
            dimension_name = "Distance"
            self.routing.AddDimension(
                transit_callback_index,
                0,
                self.max_distance_capacity,
                True,
                dimension_name,
            )
            self.distance_dimension = self.routing.GetDimensionOrDie(dimension_name)

            # Balances the distance between each vehicle.
            self.distance_dimension.SetGlobalSpanCostCoefficient(
                self.distance_global_span_cost_coefficient
            )

    def set_capacity_dimension(self):
        """Set the capacity dimension and constraint for the problem."""

        demand_callback_index = self.routing.RegisterUnaryTransitCallback(
            self.demand_callback
        )

        capacities = (
            getattr(self.model, "capacities", None)
            or [self.model.capacity] * self.model.num_vehicles
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

        for trip in self.model.trips:
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
        search_parameters.first_solution_strategy = self.solution_strategy
        search_parameters.local_search_metaheuristic = self.local_search_metaheuristic
        search_parameters.log_search = self.track_progress
        search_parameters.time_limit.seconds = self.time_limit_seconds

        return search_parameters

    def add_dummy_depot(self):
        """
        Add a dummy depot to the distance matrix, in order to solve RPP.
        It's important to add the dummy depot after the distance matrix is set, to avoid indexing issues.
        """

        self.depot = len(self.model.distance_matrix)
        self.model.distance_matrix.append([0] * len(self.model.distance_matrix))
        for i in range(len(self.model.distance_matrix)):
            self.model.distance_matrix[i].append(0)

    def remove_unused_locations(self, model: InfiniteRPP):
        """
        Remove locations that are not used in the trips. Trips are updated to reflect the new indices.
        """

        used_locations = sorted({loc for trip in model.trips for loc in trip[:2]})
        old_to_new_index = {old: new for new, old in enumerate(used_locations)}

        # Update locations list with new indices
        model.locations[:] = [model.locations[i] for i in used_locations]

        # Update trips list with new indices
        model.trips[:] = [
            (old_to_new_index[trip[0]], old_to_new_index[trip[1]], trip[2])
            for trip in model.trips
        ]

        # Update distance matrix if provided
        if model.distance_matrix is not None:
            model.distance_matrix[:] = [
                [model.distance_matrix[i][j] for j in used_locations]
                for i in used_locations
            ]

        # Update location names if provided
        if model.location_names is not None:
            model.location_names[:] = [model.location_names[i] for i in used_locations]

    def _convert_solution(self, result: Any, local_run_time: float) -> VRPSolution:
        """Converts OR-Tools result to CVRP solution."""

        routes = []
        loads = []
        distances = []
        total_distance = 0

        for vehicle_id in range(self.model.num_vehicles):
            index = self.routing.Start(vehicle_id)
            if self.use_rpp:  # Skip dummy depot
                index = result.Value(self.routing.NextVar(index))

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

            if not self.use_rpp:
                route.append(self.manager.IndexToNode(index))
                route_loads.append(cur_load)

            routes.append(route)
            distances.append(route_distance)
            loads.append(route_loads)
            total_distance += route_distance

        return VRPSolution.from_model(
            self.model,
            result.ObjectiveValue(),
            total_distance,
            routes,
            distances,
            loads,
            run_time=self.run_time,
            local_run_time=local_run_time,
        )
