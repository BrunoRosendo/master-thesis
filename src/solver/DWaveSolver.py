from dimod import ExactCQMSolver, Sampler, SampleSet

from src.model.VRPSolution import VRPSolution
from src.model.adapter.DWaveAdapter import DWaveAdapter
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
        super().__init__(
            num_vehicles,
            capacities,
            locations,
            trips,
            use_rpp,
            track_progress,
            simplify,
        )
        self.sampler = sampler
        self.adapter = DWaveAdapter(self.qubo)

    def _solve_cvrp(self) -> any:
        """
        Solve the CVRP using Quantum Annealing implemented in DWave.
        """
        cqm = self.adapter.get_model()
        print(cqm)

        result = self.sampler.sample_cqm(cqm)
        return result

    def _convert_solution(self, result: SampleSet) -> VRPSolution:
        """
        Convert the optimizer result to a VRPSolution result.
        """

        var_dict, energy = self.build_var_dict(result)
        return self.qubo.convert_result(var_dict, energy)

    def build_var_dict(self, result: SampleSet) -> (dict[str, float], float):
        """
        Builds a dictionary of variable names and their values from the result.
        Assumes the order of energy returned by the sampler.
        It takes the simplification step into consideration.

        Returns:
            dict[str, float]: Dictionary of variable names and their values.
            float: Energy of the best solution.
        """

        try:
            solution = result.filter(lambda s: s.is_feasible).lowest().first
        except ValueError:
            raise Exception("The solution is infeasible, aborting!")

        var_dict = solution.sample
        if self.simplify:
            var_dict = self.qubo.re_add_variables(dict(var_dict))
        return var_dict, solution.energy
