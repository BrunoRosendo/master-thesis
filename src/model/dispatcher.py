from typing import Callable

from src.model.VRP import DistanceUnit, VRP
from src.model.cvrp.ConstantCVRP import ConstantCVRP
from src.model.cvrp.InfiniteCVRP import InfiniteCVRP
from src.model.cvrp.MultiCVRP import MultiCVRP
from src.model.rpp.CapacityRPP import CapacityRPP
from src.model.rpp.InfiniteRPP import InfiniteRPP
from src.solver.cost_functions import manhattan_distance


def CVRP(
    num_vehicles: int,
    locations: list[tuple[float, float]],
    capacities: int | list[int] | None,
    demands: list[int] = None,
    simplify: bool = True,
    cost_function: Callable[
        [list[tuple[float, float]], DistanceUnit], list[list[float]]
    ] = manhattan_distance,
    distance_matrix: list[list[float]] | None = None,
    location_names: list[str] | None = None,
    distance_unit: DistanceUnit = DistanceUnit.METERS,
) -> VRP:
    """
    Create a Capacitated Vehicle Routing Problem model based on the given parameters. Depot is always the first location.
    Args:
        num_vehicles: Number of vehicles.
        locations: List of coordinates for each location.
        capacities: Capacity of the vehicles (same for all or specified). If None, the model will be infinite.
        demands: List of demands for each location. Must be specified if capacities is not None.
        simplify: Whether to simplify the model, removing unnecessary variables.
        cost_function:  Function to calculate the cost between two locations.
        distance_matrix: Matrix with the distance between each pair of locations.
        location_names: List of names for each location. Used for display purposes.
        distance_unit: Unit of distance used in the cost function. Used for display purposes.

    Returns:
        CVRP model.
    """
    if capacities is None:
        return InfiniteCVRP(
            num_vehicles,
            locations,
            demands or [1] * len(locations),
            simplify,
            cost_function,
            0,
            distance_matrix,
            location_names,
            distance_unit,
        )

    if demands is None:
        raise ValueError("Demands must be specified when capacities are not None")

    if isinstance(capacities, int):
        return ConstantCVRP(
            num_vehicles,
            locations,
            demands,
            capacities,
            simplify,
            cost_function,
            0,
            distance_matrix,
            location_names,
            distance_unit,
        )

    if isinstance(capacities, list):
        return MultiCVRP(
            num_vehicles,
            locations,
            demands,
            capacities,
            cost_function,
            0,
            distance_matrix,
            location_names,
            distance_unit,
        )

    raise ValueError("Invalid capacity type")


def RPP(
    num_vehicles: int,
    locations: list[tuple[float, float]],
    capacities: int | list[int] | None,
    trips: list[tuple[int, int, int]],
    cost_function: Callable[
        [list[tuple[float, float]], DistanceUnit], list[list[float]]
    ] = manhattan_distance,
    distance_matrix: list[list[float]] | None = None,
    location_names: list[str] | None = None,
    distance_unit: DistanceUnit = DistanceUnit.METERS,
) -> VRP:
    """
    Create a Ride Pooling Problem model based on the given parameters.

    Args:
        num_vehicles: Number of vehicles.
        locations: List of coordinates for each location.
        capacities: Capacity of the vehicles (same for all or specified). If None, the model will be infinite.
        trips: List of trips with origin, destination, and demand.
        cost_function: Function to calculate the cost between two locations.
        distance_matrix: Matrix of distances between locations.
        location_names: Names of the locations for display purposes.
        distance_unit: Unit of distance used in the cost function.

    Returns:
        An RPP model.
    """

    if capacities is None:
        return InfiniteRPP(
            num_vehicles,
            locations,
            trips,
            cost_function,
            distance_matrix,
            location_names,
            distance_unit,
        )

    if isinstance(capacities, int):
        capacities = [capacities] * num_vehicles

    if isinstance(capacities, list):
        return CapacityRPP(
            num_vehicles,
            locations,
            trips,
            capacities,
            cost_function,
            distance_matrix,
            location_names,
            distance_unit,
        )

    raise ValueError("Invalid capacity type")
