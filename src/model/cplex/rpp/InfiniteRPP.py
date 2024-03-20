import numpy as np
from qiskit_optimization import QuadraticProgram

from src.model.cplex.CplexVRP import CplexVRP


class InfiniteRPP(CplexVRP):
    """
    A class to represent a CPLEX math formulation of the RPP model with an infinite capacity.
    Note that this model assumes a starting point with no cost to the first node for each vehicle,
    as to avoid the need for a depot.

    Attributes:
        num_trips (int): Number of trips to be made.
        num_steps (int): Number of max steps for each vehicle.
        used_locations_indices (list): Indices of the locations used in the problem, based on trip requests.
        num_used_locations (int): Number of locations used in the problem.
    """

    def __init__(
        self,
        num_vehicles: int,
        trips: list[tuple[int, int, int]],
        distance_matrix: list[list[int]],
        locations: list[tuple[int, int]],
        simplify: bool,
    ):
        self.trips = trips
        self.num_trips = len(trips)
        self.num_steps = 2 * self.num_trips + 1

        self.used_locations_indices = self.get_used_locations()
        self.num_used_locations = len(self.used_locations_indices)

        self.epsilon = 0  # TODO Check this value
        self.normalization_factor = np.max(distance_matrix) + self.epsilon

        super().__init__(
            num_vehicles, trips, None, distance_matrix, locations, True, simplify
        )

    def create_vars(self):
        """
        Create the variables for the CPLEX model. The indices for the variables are:
        k: vehicle
        i: location (including the starting point)
        s: step
        """

        self.x = self.cplex.binary_var_cube(
            self.num_vehicles, self.num_used_locations + 1, self.num_steps, name="x"
        )

    def create_objective(self):
        """
        Create the objective function for the CPLEX model. The cost ignores the starting point.
        The trip incentive is subtracted from the cost.
        """

        cost = self.cplex.sum(
            self.distance_matrix[self.used_locations_indices[i - 1]][
                self.used_locations_indices[j - 1]
            ]
            / self.normalization_factor
            * self.x[k, i, s]
            * self.x[k, j, s + 1]
            for k in range(self.num_vehicles)
            for i in range(1, self.num_used_locations + 1)
            for j in range(1, self.num_used_locations + 1)
            for s in range(self.num_steps - 1)
        )

        trip_incentive = self.create_trip_incentive()

        self.cplex.minimize(cost - trip_incentive)

    def create_constraints(self):
        """
        Create the constraints for the CPLEX model.
        """

        self.create_location_constraints()
        self.create_vehicle_constraints()

    def create_location_constraints(self):
        """
        Create the constraints that ensure each location is visited exactly once.
        """

        for i in range(1, self.num_used_locations + 1):
            self.cplex.add_constraint(
                self.cplex.sum(
                    self.x[k, i, s]
                    for k in range(self.num_vehicles)
                    for s in range(self.num_steps)
                )
                == 1
            )

    def create_vehicle_constraints(self):
        """
        Create the constraints that ensure each vehicle can only be at one location at a time.
        """

        final_step = self.num_steps - 1

        for k in range(self.num_vehicles):
            for s in range(final_step):
                self.cplex.add_constraint(
                    self.cplex.sum(
                        self.x[k, i, s] for i in range(self.num_used_locations + 1)
                    )
                    == 1
                )

            # Half-hot constraint meant to reduce variables in the last step
            self.cplex.add_constraint(
                self.cplex.sum(
                    self.x[k, i, final_step]
                    for i in range(1, self.num_used_locations + 1)
                )
                <= 1
            )

    def create_trip_incentive(self):
        """
        Creates incentives added to the objective function to encourage the vehicles to complete the trips.
        This is the causality condition and the same as creating the complement constraints,
        which would require more variables.
        """

        return self.cplex.sum(
            self.x[k, i, s1] * self.x[k, j, s2]
            for k in range(self.num_vehicles)
            for i, j, _ in self.trips
            for s1 in range(self.num_steps - 1)
            for s2 in range(s1 + 1, self.num_steps)
        )

    def simplify_problem(self, qp: QuadraticProgram) -> QuadraticProgram:
        """
        Simplify the problem by removing unnecessary variables.
        """

        # TODO in the article

        return qp

    def get_used_locations(self) -> list[int]:
        """
        Get the indices of the locations used in the problem. This helps reduce the number of variables.
        """

        return list({location for trip in self.trips for location in trip})

    def get_result_route_starts(self, var_dict: dict[str, float]) -> list[int]:
        """
        Get the starting location for each route from the variable dictionary.
        """

        return []

    def get_result_next_location(
        self, var_dict: dict[str, float], cur_location: int
    ) -> int | None:
        """
        Get the next location for a route from the variable dictionary.
        """

        return 0

    def get_var_name(self, i: int, j: int, k: int | None = None) -> str:
        """
        Get the name of a variable.
        """

        return f"x_{i}_{j}_{k}"
