import json
import math
import os
import urllib.request

from src.model.VRPSolution import DistanceUnit


def manhattan_distance(
    locations: list[tuple[int, int]], unit: DistanceUnit = DistanceUnit.METERS
) -> list[list[float]]:
    """
    Compute the Manhattan distance between all locations.
    """

    return [
        [
            abs(from_location[0] - to_location[0])
            + abs(from_location[1] - to_location[1])
            for to_location in locations
        ]
        for from_location in locations
    ]


def euclidean_distance(
    locations: list[tuple[int, int]], unit: DistanceUnit = DistanceUnit.METERS
) -> list[list[float]]:
    """
    Compute the Euclidean distance between all locations.
    """

    return [
        [
            math.sqrt(
                (from_location[0] - to_location[0]) ** 2
                + (from_location[1] - to_location[1]) ** 2
            )
            for to_location in locations
        ]
        for from_location in locations
    ]


def distance_api(
    locations: list[tuple[int, int]], unit: DistanceUnit = DistanceUnit.METERS
) -> list[list[float]]:
    """
    Compute the distance between all locations using Google Distance API.
    Uses code extracted from https://developers.google.com/optimization/routing/vrp#distance_matrix_api.
    """

    def build_address_str(addresses: list[tuple[float, float]]):
        """Build a pipe-separated string of addresses"""

        return "|".join(f"{address[0]},{address[1]}" for address in addresses)

    def send_request(
        origin_addresses: list[tuple[float, float]],
        dest_addresses: list[tuple[float, float]],
        api_key: str,
    ) -> dict:
        """Build and send request for the given origin and destination addresses."""
        base_url = "https://maps.googleapis.com/maps/api/distancematrix/json?"
        origin_address_str = build_address_str(origin_addresses)
        dest_address_str = build_address_str(dest_addresses)
        request_url = f"{base_url}origins={origin_address_str}&destinations={dest_address_str}&key={api_key}"

        try:
            with urllib.request.urlopen(request_url) as res:
                json_result = res.read()
            return json.loads(json_result)
        except Exception as e:
            print(f"Error during request: {e}")
            return {}

    def build_distance_matrix(res: dict):
        distance_matrix = []
        for row in res.get("rows", []):
            unit_str = "distance" if unit == DistanceUnit.METERS else "duration"
            row_list = [
                row["elements"][j][unit_str]["value"]
                for j in range(len(row["elements"]))
            ]
            distance_matrix.append(row_list)
        return distance_matrix

    def fetch_and_build_matrix(
        locations: list[tuple[float, float]],
        dest_addresses: list[tuple[float, float]],
        start_idx: int,
        end_idx: int,
    ) -> list[list[float]]:
        """Fetch distances and build matrix for a range of origin addresses."""
        origin_addresses = locations[start_idx:end_idx]
        response = send_request(origin_addresses, dest_addresses, api_key)
        return build_distance_matrix(response)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Set the GOOGLE_API_KEY environment variable."
        )

    max_elements = 100
    num_locations = len(locations)
    num_requests, remaining_rows = divmod(num_locations, max_elements)
    dest_addresses = locations
    distance_matrix = []

    for i in range(num_requests):
        start_idx = i * max_elements
        end_idx = (i + 1) * max_elements
        distance_matrix += fetch_and_build_matrix(
            locations, dest_addresses, start_idx, end_idx
        )

    if remaining_rows > 0:
        start_idx = num_requests * max_elements
        end_idx = num_requests * max_elements + remaining_rows
        distance_matrix += fetch_and_build_matrix(
            locations, dest_addresses, start_idx, end_idx
        )

    return distance_matrix
