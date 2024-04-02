from dimod import ExactCQMSolver, Sampler

from src.model.VRPSolution import VRPSolution
from src.model.dwave.DWaveVRP import DWaveVRP
from src.model.dwave.cvrp.DWaveConstantCVRP import DWaveConstantCVRP
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
        cqm = self.model.constrained_quadratic_model()
        print(cqm)

        result = self.sampler.sample_cqm(cqm)
        print(result)
        return result

    def _convert_solution(self, result: any) -> VRPSolution:
        pass

    def get_model(self) -> DWaveVRP:
        return DWaveConstantCVRP(
            self.num_vehicles,
            self.distance_matrix,
            self.locations,
            self.simplify,
            self.capacities,
        )
