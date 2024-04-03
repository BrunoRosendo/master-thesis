from dimod import ExactCQMSolver, Sampler, SampleSet

from src.model.VRPSolution import VRPSolution
from src.model.dwave.DWaveVRP import DWaveVRP
from src.model.dwave.cvrp.DWaveConstantCVRP import DWaveConstantCVRP
from src.model.dwave.cvrp.DWaveInfiniteCVRP import DWaveInfiniteCVRP
from src.model.dwave.cvrp.DWaveMultiCVRP import DWaveMultiCVRP
from src.model.dwave.rpp.DWaveCapacityRPP import DWaveCapacityRPP
from src.model.dwave.rpp.DWaveInfiniteRPP import DWaveInfiniteRPP
from src.solver.VRPSolver import VRPSolver

DEFAULT_SAMPLER = ExactCQMSolver()


class DWaveSolver(VRPSolver):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with Quantum Annealing, using DWave's system.

    Attributes:
    - simplify (bool): Whether to simplify the problem by removing unnecessary constraints.
    - sampler (Sampler): The Qiskit sampler to use for the QUBO problem.
    - classic_optimizer (Optimizer): The Qiskit optimizer to use for the QUBO problem.
    - warm_start (bool): Whether to use a warm start for the QAOA optimizer.
    - pre_solver (OptimizationAlgorithm): The Qiskit optimizer to use for the pre-solver.
    - sampler (Sampler): The DWave sampler to use for the QUBO problem. Must implement the `sample_cqm` method.
    """

    def __init__(
        self,
        num_vehicles: int,
        capacities: int | list[int] | None,
        locations: list[tuple[int, int]],
        trips: list[tuple[int, int, int]],
        use_rpp: bool,
        simplify=True,
        track_progress=True,
        sampler: Sampler = DEFAULT_SAMPLER,
    ):
        self.simplify = simplify
        super().__init__(
            num_vehicles, capacities, locations, trips, use_rpp, track_progress
        )
        self.sampler = sampler

    def _solve_cvrp(self) -> any:
        """
        Solve the CVRP using Quantum Annealing implemented in DWave.
        """
        cqm = self.model.constrained_quadratic_model()
        print(cqm)

        result = self.sampler.sample_cqm(cqm)
        return result

    def _convert_solution(self, result: SampleSet) -> VRPSolution:
        """
        Convert the optimizer result to a VRPSolution result.
        """

        var_dict, energy = self.model.build_var_dict(result)
        route_starts = self.model.get_result_route_starts(var_dict)

        routes = []
        loads = []
        distances = []
        total_distance = 0

        for i in range(self.num_vehicles):
            route = []
            route_loads = []
            route_distance = 0
            cur_load = 0

            index = route_starts[i] if i < len(route_starts) else None
            previous_index = index if self.use_rpp else self.depot
            if not self.use_rpp:
                route.append(self.depot)
                route_loads.append(0)

            while index is not None:
                route_distance += self.distance_matrix[previous_index][index]
                cur_load += self.model.get_location_demand(index)
                route.append(index)
                route_loads.append(cur_load)

                if index == self.depot and not self.use_rpp:
                    break

                previous_index = index
                index = self.model.get_result_next_location(var_dict, index)

            routes.append(route)
            distances.append(route_distance)
            loads.append(route_loads)
            total_distance += route_distance

        return VRPSolution(
            self.num_vehicles,
            self.locations,
            energy,
            total_distance,
            routes,
            distances,
            self.depot,
            not self.use_rpp,
            self.capacities,
            loads if self.use_capacity else None,
        )

    def get_model(self) -> DWaveVRP:
        """
        Get an instance of the DWaveVRP model.
        """

        if self.use_rpp:
            if self.use_capacity:
                return DWaveCapacityRPP(
                    self.num_vehicles,
                    self.trips,
                    self.distance_matrix,
                    self.locations,
                    (
                        [self.capacities] * self.num_vehicles
                        if self.same_capacity
                        else self.capacities
                    ),
                )
            return DWaveInfiniteRPP(
                self.num_vehicles,
                self.trips,
                self.distance_matrix,
                self.locations,
            )

        if not self.use_capacity:
            return DWaveInfiniteCVRP(
                self.num_vehicles, self.distance_matrix, self.locations, self.simplify
            )
        if self.same_capacity:
            return DWaveConstantCVRP(
                self.num_vehicles,
                self.distance_matrix,
                self.locations,
                self.simplify,
                self.capacities,
            )

        return DWaveMultiCVRP(
            self.num_vehicles, self.distance_matrix, self.capacities, self.locations
        )
