import math
import os


def manhattan_distance(locations: list[tuple[int, int]]) -> list[list[float]]:
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


def euclidean_distance(locations: list[tuple[int, int]]) -> list[list[float]]:
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


def distance_api(locations: list[tuple[int, int]]) -> list[list[float]]:
    """
    Compute the distance between all locations using Google Distance API.
    """

    data = {}
    data["API_key"] = os.getenv("GOOGLE_API_KEY")

    return []
