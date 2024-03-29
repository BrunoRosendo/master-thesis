from docplex.util.status import JobSolveStatus
from numpy import ndarray
from qiskit.primitives import Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, Optimizer
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

from src.model.VRPSolution import VRPSolution
from src.model.cplex.CplexVRP import CplexVRP
from src.model.cplex.cvrp.ConstantCVRP import ConstantCVRP
from src.model.cplex.cvrp.InfiniteCVRP import InfiniteCVRP
from src.model.cplex.cvrp.MultiCVRP import MultiCVRP
from src.model.cplex.rpp.CapacityRPP import CapacityRPP
from src.model.cplex.rpp.InfiniteRPP import InfiniteRPP
from src.solver.VRPSolver import VRPSolver

DEFAULT_SAMPLER = Sampler()
DEFAULT_CLASSIC_OPTIMIZER = COBYLA()
DEFAULT_PRE_SOLVER = CplexOptimizer()


class QuboSolver(VRPSolver):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with QUBO algorithm, using Qiskit.

    Attributes:
    - classical_solver (bool): Whether to use a classical solver to solve the QUBO problem.
    - simplify (bool): Whether to simplify the problem by removing unnecessary constraints.
    - sampler (Sampler): The Qiskit sampler to use for the QUBO problem.
    - classic_optimizer (Optimizer): The Qiskit optimizer to use for the QUBO problem.
    - warm_start (bool): Whether to use a warm start for the QAOA optimizer.
    - pre_solver (OptimizationAlgorithm): The Qiskit optimizer to use for the pre-solver.
    """

    def __init__(
        self,
        num_vehicles: int,
        capacities: int | list[int] | None,
        locations: list[tuple[int, int]],
        trips: list[tuple[int, int, int]],
        use_rpp: bool,
        classical_solver=False,
        simplify=True,
        track_progress=True,
        sampler: Sampler = DEFAULT_SAMPLER,
        classic_optimizer: Optimizer = DEFAULT_CLASSIC_OPTIMIZER,
        warm_start=False,
        pre_solver: OptimizationAlgorithm = DEFAULT_PRE_SOLVER,
    ):
        self.simplify = simplify
        super().__init__(
            num_vehicles, capacities, locations, trips, use_rpp, track_progress
        )
        self.classical_solver = classical_solver
        self.sampler = sampler
        self.classic_optimizer = classic_optimizer
        self.warm_start = warm_start
        self.pre_solver = pre_solver

    def _solve_cvrp(self) -> OptimizationResult:
        """
        Solve the CVRP using QUBO implemented in Qiskit.
        """
        qp = self.model.quadratic_program()

        if self.classical_solver:
            print(f"The number of variables is {qp.get_num_vars()}")
            print(qp.prettyprint())
            result = self.solve_classic(qp)
        else:
            qubo = self.quadratic_to_qubo(qp)

            print(f"The number of variables is {qubo.get_num_vars()}")
            print(qubo.prettyprint())

            result = self.solve_qubo(qubo)

        self.check_feasibility(result)
        return result

    def quadratic_to_qubo(self, qp: QuadraticProgram) -> QuadraticProgram:
        """
        Convert the quadratic program to a QUBO problem, using the Qiskit converters.
        """

        # Convert inequality constraints
        ineq_to_eq = InequalityToEquality()
        qp_eq = ineq_to_eq.convert(qp)

        # Convert integer variables to binary variables
        int_to_bin = IntegerToBinary()
        qp_bin = int_to_bin.convert(qp_eq)

        # Convert linear equality constraints to penalty terms
        lin_eq_to_penalty = LinearEqualityToPenalty()
        qubo = lin_eq_to_penalty.convert(qp_bin)

        return qubo

    def solve_classic(self, qp: QuadraticProgram) -> OptimizationResult:
        """
        Solve the QUBO problem using the classical CPLEX optimizer.
        """

        optimizer = CplexOptimizer(disp=self.track_progress)
        result = optimizer.solve(qp)
        return result

    def solve_qubo(self, qp: QuadraticProgram) -> OptimizationResult:
        """
        Solve the QUBO problem using the configured Qiskit optimizer.
        """

        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=self.classic_optimizer,
            callback=self.qaoa_callback if self.track_progress else None,
        )

        if self.warm_start:
            optimizer = WarmStartQAOAOptimizer(
                pre_solver=self.pre_solver, relax_for_pre_solver=False, qaoa=qaoa
            )
        else:
            optimizer = MinimumEigenOptimizer(qaoa)

        result = optimizer.solve(qp)
        return result

    def qaoa_callback(
        self, iter_num: int, ansatz: ndarray, objective: float, metadata: dict[str, any]
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

        var_dict = self.model.build_var_dict(result)
        if not self.model.is_result_feasible(var_dict):
            raise Exception("The solution is infeasible, aborting!")

    def _convert_solution(self, result: OptimizationResult) -> VRPSolution:
        """
        Convert the optimizer result to a VRPSolution result.
        """
        var_dict = self.model.build_var_dict(result)
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
            result.fval,
            total_distance,
            routes,
            distances,
            self.depot,
            not self.use_rpp,
            self.capacities,
            loads if self.use_capacity else None,
        )

    def get_model(self) -> CplexVRP:
        """
        Get a cplex instance of the CVRPModel.
        """

        if self.use_rpp:
            if self.use_capacity:
                return CapacityRPP(
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
            return InfiniteRPP(
                self.num_vehicles,
                self.trips,
                self.distance_matrix,
                self.locations,
            )

        if not self.use_capacity:
            return InfiniteCVRP(
                self.num_vehicles, self.distance_matrix, self.locations, self.simplify
            )
        if self.same_capacity:
            return ConstantCVRP(
                self.num_vehicles,
                self.distance_matrix,
                self.capacities,
                self.locations,
                self.simplify,
            )

        return MultiCVRP(
            self.num_vehicles, self.distance_matrix, self.capacities, self.locations
        )
