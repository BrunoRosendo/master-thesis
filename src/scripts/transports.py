from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src.solver.ClassicSolver import ClassicSolver

load_dotenv()

# FILE CONSTANTS

DATA_FOLDER = "data/"
DATA_INSTANCE = "Porto/metro 04-24"
DATA_PATH = DATA_FOLDER + DATA_INSTANCE

# LOAD DATA

stops = pd.read_csv(f"{DATA_PATH}/stops.txt", sep=",")
stop_times = pd.read_csv(f"{DATA_PATH}/stop_times.txt", sep=",")
trips = pd.read_csv(f"{DATA_PATH}/trips.txt", sep=",")
routes = pd.read_csv(f"{DATA_PATH}/routes.txt", sep=",")

# CLEAN DATA

stops = stops.loc[(stops.stop_lat != 0) & (stops.stop_lon != 0)]
routes = routes.loc[
    routes.route_id != "Bexp"
]  # Express route is just a subset of the regular route

# SETUP ALGORITHM

locations = [(stop.stop_lat, stop.stop_lon) for stop in stops.itertuples()]
location_names = [stop.stop_name for stop in stops.itertuples()]

distance_matrix = [
    [0 if i == j else 999999999 for i in range(len(locations))]
    for j in range(len(locations))
]  # Initialize with large values, indicating no connection

cvrp_trips = []


def calculate_distance_matrix(trip_id, both_directions=False):
    route_stop_times = stop_times.loc[stop_times.trip_id == trip_id]

    for i in range(len(route_stop_times) - 1):
        from_stop = route_stop_times.iloc[i]
        to_stop = route_stop_times.iloc[i + 1]

        # Parse the time strings into datetime objects
        from_time = datetime.strptime(from_stop.departure_time, "%H:%M:%S")
        to_time = datetime.strptime(to_stop.arrival_time, "%H:%M:%S")

        # Calculate the difference in seconds
        distance = int((to_time - from_time).total_seconds())

        from_stop_index = stops.loc[stops.stop_id == from_stop.stop_id].index[0]
        to_stop_index = stops.loc[stops.stop_id == to_stop.stop_id].index[0]

        distance_matrix[from_stop_index][to_stop_index] = distance
        if both_directions:
            distance_matrix[to_stop_index][from_stop_index] = distance


def calculate_trips(route_id, trip_id, direction_id):
    num_trips = trips.loc[
        (trips.route_id == route_id) & (trips.direction_id == direction_id)
    ].shape[0]

    route_stop_times = stop_times.loc[stop_times.trip_id == trip_id]
    for i in range(len(route_stop_times) - 1):
        from_stop = route_stop_times.iloc[i]
        to_stop = route_stop_times.iloc[i + 1]

        from_stop_index = stops.loc[stops.stop_id == from_stop.stop_id].index[0]
        to_stop_index = stops.loc[stops.stop_id == to_stop.stop_id].index[0]

        cvrp_trips.append((from_stop_index, to_stop_index, num_trips))


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


for route in routes.itertuples():
    departure_trip_id = get_trip_id(route.route_id, 0)
    return_trip_id = get_trip_id(route.route_id, 1)

    if departure_trip_id is None and return_trip_id is None:
        raise ValueError(f"No trips found for route {route.route_id}")

    if departure_trip_id is not None:
        calculate_distance_matrix(
            departure_trip_id, both_directions=return_trip_id is None
        )
        calculate_trips(route.route_id, departure_trip_id, 0)

    if return_trip_id is not None:
        calculate_distance_matrix(
            return_trip_id, both_directions=departure_trip_id is None
        )
        if departure_trip_id is None:
            calculate_trips(route.route_id, return_trip_id, 1)


# RUN ALGORITHM

cvrp = ClassicSolver(
    1,
    None,
    locations,
    cvrp_trips,
    True,
    distance_matrix=distance_matrix,
    location_names=location_names,
)

result = cvrp.solve()
result.display()
