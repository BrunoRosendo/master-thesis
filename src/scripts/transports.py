from collections import Counter
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src.model.VRPSolution import DistanceUnit
from src.solver.ClassicSolver import ClassicSolver

load_dotenv()

# CONSTANTS

DATA_FOLDER = "data/"
DATA_INSTANCE = "Porto/stcp 09-23"
DATA_PATH = DATA_FOLDER + DATA_INSTANCE

SELECTED_ROUTES = ["300", "302"]
SELECTED_TRIP_COUNT = 3  # Only 1 in SELECTED_TRIP_COUNT trips will be added to the model. ALSO REMOVES THE STOPS
CIRCULAR_ROUTES = True
NUM_VEHICLES = 1
VEHICLE_CAPACITY = None

# LOAD DATA

stops = pd.read_csv(f"{DATA_PATH}/stops.txt", sep=",")
stop_times = pd.read_csv(f"{DATA_PATH}/stop_times.txt", sep=",")
trips = pd.read_csv(f"{DATA_PATH}/trips.txt", sep=",")
routes = pd.read_csv(f"{DATA_PATH}/routes.txt", sep=",")

# CLEAN DATA

stops = stops.loc[(stops.stop_lat != 0) & (stops.stop_lon != 0)]
routes = routes.loc[
    (routes.route_id.isin(SELECTED_ROUTES)) & (routes.route_id != "Bexp")
]  # Express route is just a subset of the regular route

# SETUP ALGORITHM


def initialize_distance_matrix(size):
    return [[0 if i == j else 999999999 for i in range(size)] for j in range(size)]


def get_time_difference(from_time_str, to_time_str):
    from_time = datetime.strptime(from_time_str, "%H:%M:%S")
    to_time = datetime.strptime(to_time_str, "%H:%M:%S")
    return int((to_time - from_time).total_seconds())


def calculate_distance_matrix(
    distance_matrix, trip_id, both_directions=False, location_ids=None
):
    route_stop_times = stop_times.loc[stop_times.trip_id == trip_id]
    from_stop = route_stop_times.iloc[0]
    count = 1

    for i in range(1, len(route_stop_times)):
        if count < SELECTED_TRIP_COUNT and i < len(route_stop_times) - 1:
            count += 1
            continue

        to_stop = route_stop_times.iloc[i]
        distance = get_time_difference(from_stop.departure_time, to_stop.arrival_time)

        if location_ids is not None:
            from_stop_indices = [
                index
                for index, loc_id in enumerate(location_ids)
                if loc_id == from_stop.stop_id
            ]
            to_stop_indices = [
                index
                for index, loc_id in enumerate(location_ids)
                if loc_id == to_stop.stop_id
            ]
        else:
            from_stop_indices = [stops.loc[stops.stop_id == from_stop.stop_id].index[0]]
            to_stop_indices = [stops.loc[stops.stop_id == to_stop.stop_id].index[0]]

        for from_idx in from_stop_indices:
            for to_idx in to_stop_indices:
                distance_matrix[from_idx][to_idx] = distance
                if both_directions:
                    distance_matrix[to_idx][from_idx] = distance

        from_stop = to_stop
        count = 1


def calculate_trips(cvrp_trips, route_id, trip_id, direction_id, location_freq):
    num_trips = trips.loc[
        (trips.route_id == route_id) & (trips.direction_id == direction_id)
    ].shape[0]

    route_stop_times = stop_times.loc[stop_times.trip_id == trip_id]
    from_stop = route_stop_times.iloc[0]
    location_freq[stops.loc[stops.stop_id == from_stop.stop_id].index[0]] += 1
    count = 1

    for i in range(1, len(route_stop_times)):
        if count < SELECTED_TRIP_COUNT and i < len(route_stop_times) - 1:
            count += 1
            continue

        to_stop = route_stop_times.iloc[i]

        from_stop_index = stops.loc[stops.stop_id == from_stop.stop_id].index[0]
        to_stop_index = stops.loc[stops.stop_id == to_stop.stop_id].index[0]

        cvrp_trips.append((int(from_stop_index), int(to_stop_index), num_trips))

        from_stop = to_stop
        location_freq[to_stop_index] += 1
        count = 1


def calculate_circular_route(locations, location_names, location_ids, trip_id):
    route_stop_times = stop_times.loc[stop_times.trip_id == trip_id]
    count = SELECTED_TRIP_COUNT

    for i in range(len(route_stop_times)):
        if count < SELECTED_TRIP_COUNT and i < len(route_stop_times) - 1:
            count += 1
            continue

        stop_time = route_stop_times.iloc[i]
        stop = stops.loc[stops.stop_id == stop_time.stop_id].iloc[0]

        if i == len(route_stop_times) - 1 and stop.stop_id in location_ids:
            break  # Circular route, last stop is the same as the first (USUALLY)

        locations.append((stop.stop_lat, stop.stop_lon))
        location_names.append(stop.stop_name)
        location_ids.append(stop.stop_id)

        count = 1


def get_trip_id(route_id, direction_id):
    try:
        return (
            trips.loc[
                (trips.route_id == route_id) & (trips.direction_id == direction_id)
            ]
            .iloc[0]
            .trip_id
        )
    except IndexError:
        return None


def check_common_stops_and_change_depot(locations, location_names, location_ids):
    location_freqs = Counter(location_ids)
    most_common_id, most_common_freq = location_freqs.most_common(1)[0]

    if most_common_freq < 2:
        raise ValueError(
            "Detected multiple lines without common stops. This is not possible for circular routes!"
        )

    if most_common_freq < len(SELECTED_ROUTES):
        print(
            "There isn't a common stop between all selected lines. The solution will possibly be unfeasible or use less"
            " vehicles than desired"
        )

    new_depot_idx = location_ids.index(most_common_id)
    locations.insert(0, locations.pop(new_depot_idx))
    location_names.insert(0, location_names.pop(new_depot_idx))
    location_ids.insert(0, location_ids.pop(new_depot_idx))


if CIRCULAR_ROUTES:
    locations = []
    location_names = []
    location_ids = []
    cvrp_trips = []

    for route in routes.itertuples():
        circular_trip_id = get_trip_id(route.route_id, 0)

        if circular_trip_id is None:
            raise ValueError(f"No trips found for route {route.route_id}")

        calculate_circular_route(
            locations, location_names, location_ids, circular_trip_id
        )

    if len(SELECTED_ROUTES) > 1:
        check_common_stops_and_change_depot(locations, location_names, location_ids)

    distance_matrix = initialize_distance_matrix(len(locations))

    for route in routes.itertuples():
        circular_trip_id = get_trip_id(route.route_id, 0)
        calculate_distance_matrix(
            distance_matrix,
            circular_trip_id,
            location_ids=location_ids,
            both_directions=True,
        )
else:
    locations = [(stop.stop_lat, stop.stop_lon) for stop in stops.itertuples()]
    location_names = [stop.stop_name for stop in stops.itertuples()]
    location_freq = [0 for _ in locations]
    distance_matrix = initialize_distance_matrix(len(locations))

    cvrp_trips = []

    for route in routes.itertuples():
        departure_trip_id = get_trip_id(route.route_id, 0)
        return_trip_id = get_trip_id(route.route_id, 1)

        if departure_trip_id is None and return_trip_id is None:
            raise ValueError(f"No trips found for route {route.route_id}")

        if departure_trip_id is not None:
            calculate_distance_matrix(
                distance_matrix,
                departure_trip_id,
                both_directions=return_trip_id is None,
            )
            calculate_trips(
                cvrp_trips, route.route_id, departure_trip_id, 0, location_freq
            )

        if return_trip_id is not None:
            calculate_distance_matrix(
                distance_matrix,
                return_trip_id,
                both_directions=departure_trip_id is None,
            )
            if departure_trip_id is None:
                calculate_trips(
                    cvrp_trips, route.route_id, return_trip_id, 1, location_freq
                )

    if any(freq > 1 for freq in location_freq):
        raise ValueError("RPP cannot handle lines with common stops")

# RUN ALGORITHM

cvrp = ClassicSolver(
    NUM_VEHICLES,
    VEHICLE_CAPACITY,
    locations,
    cvrp_trips,
    not CIRCULAR_ROUTES,
    distance_matrix=distance_matrix,
    location_names=location_names,
    distance_unit=DistanceUnit.SECONDS,
)

result = cvrp.solve()
# result.save_json("300+302-1v")
result.display()
