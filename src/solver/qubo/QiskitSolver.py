import os
from typing import Any, Literal

from docplex.util.status import JobSolveStatus
from numpy import ndarray
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA, Optimizer
from qiskit_ibm_runtime import (
    QiskitRuntimeService,
    Session,
    SamplerV1 as CloudSampler,
    Options,
)
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (
    CplexOptimizer,
    OptimizationResult,
    WarmStartQAOAOptimizer,
    MinimumEigenOptimizer,
    OptimizationAlgorithm,
    OptimizationResultStatus,
)
from qiskit_optimization.converters import (
    InequalityToEquality,
    IntegerToBinary,
    LinearEqualityToPenalty,
)

from src.model.VRP import VRP
from src.model.VRPSolution import VRPSolution
from src.model.adapter.QiskitAdapter import QiskitAdapter
from src.qiskit_algorithms.qiskit_algorithms import QAOA
from src.solver.qubo.QuboSolver import QuboSolver

DEFAULT_SAMPLER = Sampler()
DEFAULT_CLASSICAL_OPTIMIZER = COBYLA()
DEFAULT_PRE_SOLVER = CplexOptimizer()


class QiskitSolver(QuboSolver):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with QUBO algorithm,
    using CPLEX or Qiskit.

    Attributes:
    - classical_solver (bool): Whether to use a classical solver to solve the QUBO problem.
    - sampler (Sampler): The Qiskit sampler to use for the QUBO problem.
    - classical_optimizer (Optimizer): The Qiskit optimizer to use for the QUBO problem.
    - warm_start (bool): Whether to use a warm start for the QAOA optimizer.
    - pre_solver (OptimizationAlgorithm): The Qiskit optimizer to use for the pre-solver.
    - adapter (CplexAdapter): The adapter to convert the model to a Qiskit QuadraticProgram.
    - var_dict (dict[str, float]): The dictionary with the variable values from the result.
    """

    def __init__(
        self,
        model: VRP,
        classical_solver=False,
        track_progress=True,
        sampler: Sampler | CloudSampler = DEFAULT_SAMPLER,
        classical_optimizer: Optimizer = DEFAULT_CLASSICAL_OPTIMIZER,
        warm_start=False,
        pre_solver: OptimizationAlgorithm = DEFAULT_PRE_SOLVER,
    ):
        super().__init__(model, track_progress)
        self.classical_solver = classical_solver
        self.sampler = sampler
        self.classical_optimizer = classical_optimizer
        self.warm_start = warm_start
        self.pre_solver = pre_solver
        self.adapter = QiskitAdapter(self.model)
        self.var_dict = None

    def _solve_cvrp(self) -> OptimizationResult:
        """
        Solve the CVRP using QUBO implemented in Qiskit.
        """

        qp = self.adapter.solver_model()

        if self.classical_solver:
            print(f"The number of variables is {qp.get_num_vars()}")
            print(qp.prettyprint())
            result = self.solve_classic(qp)
        else:
            qp = self.convert_quadratic_program(qp)

            print(f"The number of variables is {qp.get_num_vars()}")
            print(qp.prettyprint())

            result = self.solve_qaoa(qp)

        self.check_feasibility(result)
        return result

    @staticmethod
    def convert_quadratic_program(qp: QuadraticProgram) -> QuadraticProgram:
        """
        Convert the quadratic program to a canonic formulation, using the Qiskit converters.
        """

        # Convert inequality constraints
        ineq_to_eq = InequalityToEquality()
        qp_eq = ineq_to_eq.convert(qp)

        # Convert integer variables to binary variables
        int_to_bin = IntegerToBinary()
        qp_bin = int_to_bin.convert(qp_eq)

        # Convert linear equality constraints to penalty terms
        lin_eq_to_penalty = LinearEqualityToPenalty()
        qp_penalties = lin_eq_to_penalty.convert(qp_bin)

        return qp_penalties

    def solve_classic(self, qp: QuadraticProgram) -> OptimizationResult:
        """
        Solve the QUBO problem using the classical CPLEX optimizer.
        """

        optimizer = CplexOptimizer(disp=self.track_progress)
        result, self.run_time = self.measure_time(optimizer.solve, qp)

        return result

    def solve_qaoa(self, qp: QuadraticProgram) -> OptimizationResult:
        """
        Solve the QUBO problem using the configured Qiskit optimizer.
        """

        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=self.classical_optimizer,
            callback=self.qaoa_callback if self.track_progress else None,
        )

        if self.warm_start:
            optimizer = WarmStartQAOAOptimizer(
                pre_solver=self.pre_solver, relax_for_pre_solver=False, qaoa=qaoa
            )
        else:
            optimizer = MinimumEigenOptimizer(qaoa)

        result, self.run_time = self.measure_time(optimizer.solve, qp)
        return result

    @staticmethod
    def qaoa_callback(
        iter_num: int, ansatz: ndarray, objective: float, metadata: dict[str, Any]
    ):
        print(f"Iteration {iter_num}: {objective.real} objective")

    def check_feasibility(self, result: OptimizationResult):
        if result.status == OptimizationResultStatus.FAILURE:
            raise Exception("Failed to solve the problem, aborting!")

        if result.raw_results is None:
            if result.status != OptimizationResultStatus.SUCCESS:
                raise Exception(
                    f"Problem is not successful, aborting as {result.status.name.lower()}! (raw results not available)"
                )
        else:
            # Results marked as infeasible by the solver can still have valid solutions
            raw_status = result.raw_results.solve_status
            status_name = raw_status.name.lower().replace("_", " ")
            print(f"The solver marked the result as {status_name}")

            if raw_status in [
                JobSolveStatus.INFEASIBLE_SOLUTION,
                JobSolveStatus.UNBOUNDED_SOLUTION,
                JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION,
            ]:
                raise Exception("The problem is infeasible or unbounded, aborting!")

        self.var_dict = self.build_var_dict(result)
        if not self.model.is_result_feasible(self.var_dict):
            raise Exception("The solution is infeasible, aborting!")

    def _convert_solution(
        self, result: OptimizationResult, local_run_time: int
    ) -> VRPSolution:
        """
        Convert the optimizer result to a VRPSolution result.
        """

        if result.raw_results is not None:
            # Convert to microseconds
            self.run_time = round(result.raw_results.solve_details.time * 1e6)

        return self.convert_qubo_result(
            self.var_dict, result.fval, self.run_time, local_run_time
        )

    def build_var_dict(self, result: OptimizationResult) -> dict[str, float]:
        """
        Build a dictionary with the variable values from the result. It takes the simplification step into consideration
        """
        var_dict = result.variables_dict
        if self.model.simplify:
            var_dict = self.model.re_add_variables(var_dict)
        return var_dict


def get_backend_sampler(
    backend_name: str = None,
    channel: Literal["ibm_quantum"] | Literal["ibm_cloud"] = "ibm_quantum",
    resilience_level: int = 0,
) -> CloudSampler:
    """
    Get a Qiskit sampler for the specified backend.
    Loads the IBM-Q account using the token from the environment variable.
    """

    token = os.getenv("IBM_TOKEN")
    if not token:
        raise ValueError("IBM token not found. Set the IBM_TOKEN environment variable.")
    service = QiskitRuntimeService(token=token, channel=channel)

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(operational=True, simulator=False)

    session = Session(service, backend)

    options = Options()
    options.resilience_level = resilience_level

    return CloudSampler(session=session, options=options)
