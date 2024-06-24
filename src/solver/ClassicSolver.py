from typing import Any, Callable

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution, DistanceUnit
from src.solver.VRPSolver import VRPSolver
from src.solver.cost_functions import manhattan_distance

DEFAULT_SOLUTION_STRATEGY = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
DEFAULT_LOCAL_SEARCH_METAHEURISTIC = (
    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
)
DEFAULT_DISTANCE_GLOBAL_SPAN_COST_COEFFICIENT = 100
DEFAULT_TIME_LIMIT_SECONDS = 10


class ClassicSolver(VRPSolver):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with classic algorithms, using Google's OR Tools.

    Attributes:
    - solution_strategy (int): The strategy to use to find the first solution.
    - local_search_metaheuristic (int): The local search metaheuristic to use.
    - distance_global_span_cost_coefficient (int): The coefficient for the global span cost.
    - distance_dimension (pywrapcp.RoutingDimension): The distance dimension for the problem.
    """

    def __init__(
        self,
        num_vehicles: int,
        capacities: int | list[int] | None,
        locations: list[tuple[float, float]],
        trips: list[tuple[int, int, int]],
        use_rpp: bool,
        track_progress: bool = True,
        solution_strategy: int = DEFAULT_SOLUTION_STRATEGY,
        local_search_metaheuristic: int = DEFAULT_LOCAL_SEARCH_METAHEURISTIC,
        distance_global_span_cost_coefficient: int = DEFAULT_DISTANCE_GLOBAL_SPAN_COST_COEFFICIENT,
        time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
        cost_function: Callable[
            [list[tuple[float, float]], DistanceUnit], list[list[float]]
        ] = manhattan_distance,
        distance_matrix: list[list[float]] = None,
        location_names: list[str] = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
    ):
        if use_rpp:
            self.remove_unused_locations(
                locations, trips, distance_matrix, location_names
            )

        super().__init__(
            num_vehicles,
            capacities,
            locations,
            trips,
            use_rpp,
            track_progress,
            cost_function,
            distance_matrix=distance_matrix,
            location_names=location_names,
            distance_unit=distance_unit,
        )

        self.solution_strategy = solution_strategy
        self.local_search_metaheuristic = local_search_metaheuristic
        self.distance_global_span_cost_coefficient = (
            distance_global_span_cost_coefficient
        )
        self.time_limit_seconds = time_limit_seconds

        if self.use_rpp:
            self.add_dummy_depot()

        self.distance_dimension = None

    def _solve_cvrp(self) -> Any:
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
        return self.distance_matrix[from_node][to_node]

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
                3000,
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

        self.depot = len(self.distance_matrix)
        self.distance_matrix.append([0] * len(self.distance_matrix))
        for i in range(len(self.distance_matrix)):
            self.distance_matrix[i].append(0)

        # Update the model with the new dummy
        self.model = self.get_model()

    def remove_unused_locations(
        self,
        locations: list[tuple[float, float]],
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[float]] = None,
        location_names: list[str] = None,
    ):
        """
        Remove locations that are not used in the trips. Trips are updated to reflect the new indices.
        """

        used_locations = sorted({loc for trip in trips for loc in trip[:2]})
        old_to_new_index = {old: new for new, old in enumerate(used_locations)}

        # Update locations list with new indices
        locations[:] = [locations[i] for i in used_locations]

        # Update trips list with new indices
        trips[:] = [
            (old_to_new_index[trip[0]], old_to_new_index[trip[1]], trip[2])
            for trip in trips
        ]

        # Update distance matrix if provided
        if distance_matrix is not None:
            distance_matrix[:] = [
                [distance_matrix[i][j] for j in used_locations] for i in used_locations
            ]

        # Update location names if provided
        if location_names is not None:
            location_names[:] = [location_names[i] for i in used_locations]

    def _convert_solution(self, result: Any, local_run_time: float) -> VRPSolution:
        """Converts OR-Tools result to CVRP solution."""

        routes = []
        loads = []
        distances = []
        total_distance = 0

        for vehicle_id in range(self.num_vehicles):
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

        return VRPSolution(
            self.num_vehicles,
            self.locations,
            result.ObjectiveValue(),
            total_distance,
            routes,
            distances,
            self.depot,
            self.capacities,
            loads if self.use_capacity else None,
            not self.use_rpp,
            run_time=self.run_time,
            local_run_time=local_run_time,
            location_names=self.location_names,
            distance_unit=self.distance_unit,
        )

    def get_model(self) -> VRP:
        return VRP(
            self.num_vehicles,
            self.trips,
            self.distance_matrix,
            self.locations,
            self.use_rpp,
            self.depot,
            self.location_names,
            self.distance_unit,
        )
