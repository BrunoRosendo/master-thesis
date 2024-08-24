from typing import Callable

from src.model.StepVRP import StepVRP
from src.model.VRP import DistanceUnit
from src.solver.cost_functions import manhattan_distance


class InfiniteRPP(StepVRP):
    """
    A class to represent a QUBO math formulation of the RPP model with an infinite capacity.
    Note that this model assumes a starting point with no cost to the first node for each vehicle,
    as to avoid the need for a depot.

    This model is always be simplified, since some constraints assume the simplification.

    Attributes:
        trips (list): List of tuples, where each tuple contains the pickup and delivery locations, and the amount of passengers.
        num_trips (int): Number of trips to be made.
        used_locations_indices (list): Indices of the locations used in the problem, based on trip requests.
    """

    def __init__(
        self,
        num_vehicles: int,
        locations: list[tuple[float, float]],
        trips: list[tuple[int, int, int]],
        cost_function: Callable[
            [list[tuple[float, float]], DistanceUnit], list[list[float]]
        ] = manhattan_distance,
        distance_matrix: list[list[float]] | None = None,
        location_names: list[str] | None = None,
        distance_unit: DistanceUnit = DistanceUnit.METERS,
    ):
        self.trips = trips
        self.num_trips = len(trips)
        self.used_locations_indices = self.get_used_locations()

        super().__init__(
            num_vehicles,
            locations,
            None,
            None,
            cost_function,
            distance_matrix,
            location_names,
            distance_unit,
        )

    def create_objective(self):
        """
        Create the objective function for the RPP model. The cost ignores the starting point.
        The trip incentive is subtracted from the cost.
        """

        cost = self.model.sum(
            self.distance_matrix[self.used_locations_indices[i - 1]][
                self.used_locations_indices[j - 1]
            ]
            / self.normalization_factor
            * self.x_var(k, i, s)
            * self.x_var(k, j, s + 1)
            for k in range(self.num_vehicles)
            for i in range(1, self.num_used_locations)
            for j in range(1, self.num_used_locations)
            for s in range(self.num_steps - 1)
        )

        # We assume the weight of returning to start as 1 (max)
        return_to_start_penalty = self.model.sum(
            self.x_var(k, i, s) * self.x_var(k, 0, s + 1)
            for k in range(self.num_vehicles)
            for i in range(1, self.num_used_locations)
            for s in range(self.num_steps - 1)
        )

        trip_incentive = self.create_trip_incentive()

        return cost + return_to_start_penalty - trip_incentive

    def create_trip_incentive(self):
        """
        Creates incentives added to the objective function to encourage the vehicles to complete the trips.
        This is the causality condition and the same as creating the complement constraints,
        which would require more variables.
        """

        return self.model.sum(
            self.x_var(k, self.used_locations_indices.index(i) + 1, s1)
            * self.x_var(k, self.used_locations_indices.index(j) + 1, s2)
            for k in range(self.num_vehicles)
            for i, j, _ in self.trips
            for s1 in range(self.num_steps - 1)
            for s2 in range(s1 + 1, self.num_steps)
        )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle can only be at one location at a time.
        """

        final_step = self.num_steps - 1

        self.constraints.extend(
            self.model.sum(self.x_var(k, i, s) for i in range(self.num_used_locations))
            == 1
            for k in range(self.num_vehicles)
            for s in range(final_step)
        )

        # Half-hot constraint meant to reduce variables in the last step
        self.constraints.extend(
            self.model.sum(
                self.x_var(k, i, final_step) for i in range(1, self.num_used_locations)
            )
            <= 1
            for k in range(self.num_vehicles)
        )

    def get_simplified_variables(self) -> dict[str, int]:
        """
        Get the variables that should be replaced during the simplification and their values.
        """

        variables = {}

        for k in range(self.num_vehicles):
            # Vehicles start at their respective locations and not anywhere else
            variables[self.get_var_name(k, 0, 0)] = 1
            for i in range(1, self.num_used_locations):
                variables[self.get_var_name(k, i, 0)] = 0

            # Vehicles can't be at the starting point at the last step (see paper)
            variables[self.get_var_name(k, 0, self.num_steps - 1)] = 0

            # It's impossible to go from start to a drop-off and to end at a pick-up
            for i, j, _ in self.trips:
                variables[
                    self.get_var_name(k, self.used_locations_indices.index(j) + 1, 1)
                ] = 0
                variables[
                    self.get_var_name(
                        k, self.used_locations_indices.index(i) + 1, self.num_steps - 1
                    )
                ] = 0

        return variables

    def get_location_demand(self, idx: int) -> int:
        """
        Get the demand for a location.
        """
        pickup_demand = sum(trip[2] for trip in self.trips if idx == trip[0])
        delivery_demand = sum(trip[2] for trip in self.trips if idx == trip[1])
        return pickup_demand - delivery_demand

    def get_num_steps(self):
        """
        Get the number of steps for the model.
        """

        return 2 * self.num_trips + 1

    def get_num_used_locations(self):
        """
        Get the number of used locations for the model.
        """

        return len(self.used_locations_indices) + 1

    def get_used_locations(self) -> list[int]:
        """
        Get the indices of the locations used in the problem. This helps reduce the number of variables.
        """

        return list({location for trip in self.trips for location in trip[:2]})

    def get_result_next_location(
        self, var_dict: dict[str, float], cur_location: int
    ) -> int | None:
        """
        Get the next location for a route from the variable dictionary.
        """

        original_location = self.used_locations_indices.index(cur_location) + 1
        return super().get_result_next_location(var_dict, original_location)

    def get_result_location(
        self, var_dict: dict[str, float], k: int, s: int
    ) -> int | None:
        """
        Get the location for a vehicle at a given step.
        """

        for i, location in enumerate(self.used_locations_indices):
            if self.get_var(var_dict, k, i + 1, s) == 1.0:
                return location

        return None

    def is_result_feasible(self, var_dict: dict[str, float]) -> bool:
        """
        Check if the result is feasible by validating the trip constraints.
        This post-processing is needed because the model uses a trip incentive instead of constraints.
        However, the feasible solutions will be chosen if they exist. This removes false positives.
        """

        for i, j, _ in self.trips:
            i_var = self.used_locations_indices.index(i) + 1
            j_var = self.used_locations_indices.index(j) + 1

            trip_sum = sum(
                self.get_var(var_dict, k, i_var, s1)
                * self.get_var(var_dict, k, j_var, s2)
                for k in range(self.num_vehicles)
                for s1 in range(self.num_steps - 1)
                for s2 in range(s1 + 1, self.num_steps)
            )

            if trip_sum < 1:
                return False

        return True
