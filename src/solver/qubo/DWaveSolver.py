from logging import warning

from dimod import (
    ExactCQMSolver,
    Sampler,
    SampleSet,
    ConstrainedQuadraticModel,
    cqm_to_bqm,
    BinaryQuadraticModelStructureError,
    BinaryQuadraticModel,
)
from dimod.constrained.constrained import CQMToBQMInverter
from dwave.system import EmbeddingComposite

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution
from src.model.adapter.DWaveAdapter import DWaveAdapter
from src.solver.qubo.QuboSolver import QuboSolver

DEFAULT_SAMPLER = ExactCQMSolver()
DEFAULT_EMBEDDING = EmbeddingComposite


class DWaveSolver(QuboSolver):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with Quantum Annealing, using DWave's system.

    Attributes:
    - sampler (Sampler): The Qiskit sampler to use for the QUBO problem.
    - warm_start (bool): Whether to use a warm start for the QAOA optimizer.
    - pre_solver (OptimizationAlgorithm): The Qiskit optimizer to use for the pre-solver.
    - sampler (Sampler): The DWave sampler to use for the QUBO problem. Must implement the `sample_cqm` method.
    - embedding (ComposedSampler): The embedding to be added if a BQM sampler is used.
    - embed_bqm (bool): Whether to embed the BQM before sampling. This must be true if using a real BQM sampler
    and false otherwise.
    - num_reads (int): Number of reads for the sampler. Defaults to None, decided by the sampler.
    - adapter (DWaveAdapter): Adapter to convert the QUBO model to the DWave model.
    - cqm (ConstrainedQuadraticModel): The CQM model to be solved.
    - use_bqm (bool): Flag to indicate if the model should be converted to a BQM.
    - invert (CQMToBQMInverter): Inverter to convert the BQM solution back to the CQM solution.
    - time_limit (int): Time limit for the sampler, in seconds.
    - embedding_timeout (int): Timeout for the embedding, in seconds.
    """

    def __init__(
        self,
        model: VRP,
        track_progress=True,
        sampler: Sampler = DEFAULT_SAMPLER,
        embedding: type = DEFAULT_EMBEDDING,
        embed_bqm=True,
        embedding_timeout: int = None,
        num_reads: int = None,
        time_limit: int = None,
    ):
        super().__init__(model, track_progress)
        self.sampler = sampler
        self.embedding = embedding
        self.num_reads = num_reads
        self.time_limit = time_limit
        self.use_bqm = not self.is_cqm_sampler(sampler)
        self.adapter = DWaveAdapter(self.model, self.use_bqm)
        self.embed_bqm = embed_bqm
        self.embedding_timeout = embedding_timeout
        self.invert: CQMToBQMInverter | None = None
        self.cqm = self.adapter.solver_model()

    def _solve_cvrp(self) -> SampleSet:
        """
        Solve the CVRP using Quantum Annealing implemented in DWave.
        """
        print(self.cqm)

        if self.use_bqm:
            result, self.invert = self.sample_as_bqm(self.cqm)
        else:
            result = self.sample_cqm()

        return result

    def _convert_solution(
        self, result: SampleSet, local_run_time: float
    ) -> VRPSolution:
        """
        Convert the optimizer result to a VRPSolution result.
        """

        var_dict, energy = self.build_var_dict(result)
        timing = result.info.get("timing") or result.info
        return self.convert_qubo_result(
            var_dict,
            energy,
            timing.get("run_time") or timing.get("qpu_access_time") or self.run_time,
            local_run_time,
            timing.get("qpu_access_time"),
        )

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
            solution = result.filter(self.is_sample_feasible).lowest().first

        except ValueError:
            raise Exception("The solution is infeasible, aborting!")

        var_dict = self.invert(solution.sample) if self.use_bqm else solution.sample
        if self.model.simplify:
            var_dict = self.model.re_add_variables(dict(var_dict))
        return var_dict, solution.energy

    def is_sample_feasible(self, s):
        """
        Check if the sample is feasible (part of SampleSet).
        @type s: Sample
        """

        if self.use_bqm:
            inverted_sample = self.invert(s.sample)
            return self.cqm.check_feasible(inverted_sample)

        return s.is_feasible

    @staticmethod
    def is_cqm_sampler(sampler: Sampler) -> bool:
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
        print("Number of variables in BQM: ", len(bqm.variables))

        try:
            return self.sample_bqm(bqm, composed_sampler), invert
        except BinaryQuadraticModelStructureError:
            raise Exception(
                "The BQM structure is invalid. Make sure `embed_bqm` is set to True when running in a real BQM sampler."
            )

    def sample_cqm(self) -> SampleSet:
        """
        Sample the CQM using the selected sampler and time limit.
        """
        kwargs = {"time_limit": self.time_limit} if self.time_limit else {}
        result, self.run_time = self.measure_time(
            self.sampler.sample_cqm, self.cqm, **kwargs
        )

        return result

    def sample_bqm(
        self, bqm: BinaryQuadraticModel, sampler: Sampler = None
    ) -> SampleSet:
        """
        Sample the BQM using the selected sampler, number of reads and time limit.
        """

        sampler = sampler or self.sampler
        kwargs = {"num_reads": self.num_reads} if self.num_reads else {}
        if self.time_limit:
            kwargs["time_limit"] = self.time_limit
        if self.embed_bqm:
            kwargs["return_embedding"] = True
        if self.embedding_timeout:
            kwargs["embedding_parameters"] = dict(timeout=self.embedding_timeout)

        result, self.run_time = self.measure_time(sampler.sample, bqm, **kwargs)
        return result
