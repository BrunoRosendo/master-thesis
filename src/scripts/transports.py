from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from dwave.system import LeapHybridCQMSampler

from src.solver.qubo.DWaveSolver import DWaveSolver

load_dotenv()

# CONSTANTS

DATA_FOLDER = "data/"
DATA_INSTANCE = "Porto/stcp 09-23"
DATA_PATH = DATA_FOLDER + DATA_INSTANCE

SELECTED_ROUTES = ["18", "304"]
SELECTED_TRIP_COUNT = 1  # Only 1 in SELECTED_TRIP_COUNT trips will be added to the model. ALSO REMOVES THE STOPS
CIRCULAR_ROUTES = False

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
            from_stop_index = location_ids.index(from_stop.stop_id)
            to_stop_index = location_ids.index(to_stop.stop_id)
        else:
            from_stop_index = stops.loc[stops.stop_id == from_stop.stop_id].index[0]
            to_stop_index = stops.loc[stops.stop_id == to_stop.stop_id].index[0]

        distance_matrix[from_stop_index][to_stop_index] = distance
        if both_directions:
            distance_matrix[to_stop_index][from_stop_index] = distance

        from_stop = to_stop
        count = 1


def calculate_trips(cvrp_trips, route_id, trip_id, direction_id):
    num_trips = trips.loc[
        (trips.route_id == route_id) & (trips.direction_id == direction_id)
    ].shape[0]

    route_stop_times = stop_times.loc[stop_times.trip_id == trip_id]
    from_stop = route_stop_times.iloc[0]
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
        count = 1


def calculate_circular_route(locations, location_names, location_ids, trip_id):
    route_stop_times = stop_times.loc[stop_times.trip_id == trip_id]
    count = SELECTED_TRIP_COUNT

    for i in range(
        len(route_stop_times) - 1
    ):  # Exclude return to first stop, the algorithm will handle it
        if count < SELECTED_TRIP_COUNT and i < len(route_stop_times) - 1:
            count += 1
            continue

        stop_time = route_stop_times.iloc[i]
        stop = stops.loc[stops.stop_id == stop_time.stop_id].iloc[0]

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

    distance_matrix = initialize_distance_matrix(len(locations))

    for route in routes.itertuples():
        circular_trip_id = get_trip_id(route.route_id, 0)
        calculate_distance_matrix(
            distance_matrix, circular_trip_id, location_ids=location_ids
        )
else:
    locations = [(stop.stop_lat, stop.stop_lon) for stop in stops.itertuples()]
    location_names = [stop.stop_name for stop in stops.itertuples()]
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
            calculate_trips(cvrp_trips, route.route_id, departure_trip_id, 0)

        if return_trip_id is not None:
            calculate_distance_matrix(
                distance_matrix,
                return_trip_id,
                both_directions=departure_trip_id is None,
            )
            if departure_trip_id is None:
                calculate_trips(cvrp_trips, route.route_id, return_trip_id, 1)


# RUN ALGORITHM

cvrp = DWaveSolver(
    2,
    None,
    locations,
    cvrp_trips,
    not CIRCULAR_ROUTES,
    distance_matrix=distance_matrix,
    location_names=location_names,
    sampler=LeapHybridCQMSampler(),
    time_limit=30,
)

result = cvrp.solve()
# result.save_json("18+304")
result.display()
