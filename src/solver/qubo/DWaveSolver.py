from logging import warning

from dimod import (
    ExactCQMSolver,
    Sampler,
    SampleSet,
    ConstrainedQuadraticModel,
    cqm_to_bqm,
    BinaryQuadraticModelStructureError,
)
from dimod.constrained.constrained import CQMToBQMInverter
from dwave.system import EmbeddingComposite

from src.model.VRPSolution import VRPSolution
from src.model.adapter.DWaveAdapter import DWaveAdapter
from src.solver.qubo.QuboSolver import QuboSolver

DEFAULT_SAMPLER = ExactCQMSolver()
DEFAULT_EMBEDDING = EmbeddingComposite


class DWaveSolver(QuboSolver):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with Quantum Annealing, using DWave's system.

    Attributes:
    - simplify (bool): Whether to simplify the problem by removing unnecessary constraints.
    - sampler (Sampler): The Qiskit sampler to use for the QUBO problem.
    - classic_optimizer (Optimizer): The Qiskit optimizer to use for the QUBO problem.
    - warm_start (bool): Whether to use a warm start for the QAOA optimizer.
    - pre_solver (OptimizationAlgorithm): The Qiskit optimizer to use for the pre-solver.
    - sampler (Sampler): The DWave sampler to use for the QUBO problem. Must implement the `sample_cqm` method.
    - embedding (ComposedSampler): The embedding to be added if a BQM sampler is used.
    - embed_bqm (bool): Whether to embed the BQM before sampling. This must be true if using a real BQM sampler
    and false otherwise.
    - num_reads (int): Number of reads for the sampler. Defaults to None, decided by the sampler.
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
        embedding: type = DEFAULT_EMBEDDING,
        embed_bqm=True,
        num_reads: int = None,
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
        self.embedding = embedding
        self.num_reads = num_reads
        self.adapter = DWaveAdapter(self.model)
        self.use_bqm = not self.is_cqm_sampler(sampler)
        self.embed_bqm = embed_bqm
        self.invert: CQMToBQMInverter | None = None

    def _solve_cvrp(self) -> SampleSet:
        """
        Solve the CVRP using Quantum Annealing implemented in DWave.
        """
        cqm = self.adapter.solver_model()
        print(cqm)

        if self.use_bqm:
            result, self.invert = self.sample_as_bqm(cqm)
            return result

        return self.sampler.sample_cqm(cqm, num_reads=self.num_reads)

    def _convert_solution(self, result: SampleSet) -> VRPSolution:
        """
        Convert the optimizer result to a VRPSolution result.
        """

        var_dict, energy = self.build_var_dict(result)
        return self.model.convert_result(var_dict, energy)

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
            var_dict = self.model.re_add_variables(dict(var_dict))
        return var_dict, solution.energy

    def is_cqm_sampler(self, sampler: Sampler) -> bool:
        """
        Check if the sampler is a CQM sampler.
        """
        return hasattr(sampler, "sample_cqm")

    def sample_as_bqm(
        self, cqm: ConstrainedQuadraticModel
    ) -> (SampleSet, CQMToBQMInverter):
        """
        Sample the CQM as a BQM using the selected embedding.
        """

        try:
            composed_sampler = (
                self.embedding(self.sampler) if self.embed_bqm else self.sampler
            )
        except ValueError:
            warning(
                "Embedding failed, continuing sampler without embedding. Set `embed_bqm` to False when running locally!"
            )
            composed_sampler = self.sampler

        bqm, invert = cqm_to_bqm(cqm)

        try:
            result = (
                composed_sampler.sample(bqm)  # default value differs from None
                if self.num_reads is None
                else composed_sampler.sample(bqm, num_reads=self.num_reads)
            )
        except BinaryQuadraticModelStructureError:
            raise Exception(
                "The BQM structure is invalid. Make sure `embed_bqm` is set to True when running in a real BQM sampler."
            )

        return result, invert
