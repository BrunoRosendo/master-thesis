import math


def manhattan_distance(
    from_location: tuple[int, int], to_location: tuple[int, int]
) -> float:
    """
    Compute the Manhattan distance between two locations.
    """

    return abs(from_location[0] - to_location[0]) + abs(
        from_location[1] - to_location[1]
    )


def euclidean_distance(
    from_location: tuple[int, int], to_location: tuple[int, int]
) -> float:
    """
    Compute the Euclidean distance between two locations.
    """

    return math.sqrt(
        (from_location[0] - to_location[0]) ** 2
        + (from_location[1] - to_location[1]) ** 2
    )
