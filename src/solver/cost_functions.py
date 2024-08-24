import json
import math
import os
import urllib.request

from src.model.VRP import DistanceUnit


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
            round(
                math.sqrt(
                    (from_location[0] - to_location[0]) ** 2
                    + (from_location[1] - to_location[1]) ** 2
                )
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
        origin_addresses: list[tuple[float, float]],
        dest_addresses: list[tuple[float, float]],
    ) -> list[list[float]]:
        """Fetch distances and build matrix for a range of origin addresses."""

        response = send_request(origin_addresses, dest_addresses, api_key)
        if response["status"] != "OK":
            raise ValueError(f"Request to distance API failed: {response}")
        return build_distance_matrix(response)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Set the GOOGLE_API_KEY environment variable."
        )

    max_elements = 100
    num_locations = len(locations)
    max_origins_destinations = math.isqrt(max_elements)
    distance_matrix = [[0 for _ in range(num_locations)] for _ in range(num_locations)]

    for i in range(0, num_locations, max_origins_destinations):
        for j in range(0, num_locations, max_origins_destinations):
            origin_chunk = locations[i : i + max_origins_destinations]
            dest_chunk = locations[j : j + max_origins_destinations]
            chunk_matrix = fetch_and_build_matrix(origin_chunk, dest_chunk)

            for oi, origin in enumerate(origin_chunk):
                for di, dest in enumerate(dest_chunk):
                    distance_matrix[i + oi][j + di] = chunk_matrix[oi][di]

    return distance_matrix
