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
        use_deliveries: bool,
        use_rpp: bool,
        classical_solver=False,
        simplify=True,
        sampler: Sampler = DEFAULT_SAMPLER,
        classic_optimizer: Optimizer = DEFAULT_CLASSIC_OPTIMIZER,
        warm_start=False,
        pre_solver: OptimizationAlgorithm = DEFAULT_PRE_SOLVER,
    ):
        self.simplify = simplify
        super().__init__(
            num_vehicles, capacities, locations, trips, use_deliveries, use_rpp
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

        if result.status.name != "SUCCESS":
            raise Exception("Failed to solve the problem!")

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

        optimizer = CplexOptimizer()
        result = optimizer.solve(qp)
        return result

    def solve_qubo(self, qp: QuadraticProgram) -> OptimizationResult:
        """
        Solve the QUBO problem using the configured Qiskit optimizer.
        """

        qaoa = QAOA(sampler=self.sampler, optimizer=self.classic_optimizer)

        if self.warm_start:
            optimizer = WarmStartQAOAOptimizer(
                pre_solver=self.pre_solver, relax_for_pre_solver=False, qaoa=qaoa
            )
        else:
            optimizer = MinimumEigenOptimizer(qaoa)

        result = optimizer.solve(qp)
        return result

    def _convert_solution(self, result: OptimizationResult) -> VRPSolution:
        """
        Convert the optimizer result to a CVRPSolution solution.
        """
        var_dict = result.variables_dict
        route_starts = self.model.get_result_route_starts(var_dict)

        routes = []
        loads = []
        distances = []
        total_distance = 0

        for start in route_starts:
            index = start
            previous_index = self.depot

            route = [self.depot]
            route_loads = [0]
            route_distance = 0
            cur_load = 0

            while True:
                route_distance += self.distance_matrix[previous_index][index]
                cur_load += self.model.get_location_demand(index)
                route.append(index)
                route_loads.append(cur_load)

                if index == self.depot:
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
            self.capacities,
            loads if self.use_capacity else None,
        )

    def get_model(self) -> CplexVRP:
        """
        Get a cplex instance of the CVRPModel.
        """

        if self.use_rpp:
            return InfiniteRPP(
                self.num_vehicles,
                self.trips,
                self.distance_matrix,
                self.locations,
                self.simplify,
            )

        if not self.use_capacity:
            return InfiniteCVRP(
                self.num_vehicles,
                self.trips,
                self.depot,
                self.distance_matrix,
                self.locations,
                self.use_deliveries,
                self.simplify,
            )
        elif self.same_capacity:
            return ConstantCVRP(
                self.num_vehicles,
                self.trips,
                self.depot,
                self.distance_matrix,
                self.capacities,
                self.locations,
                self.use_deliveries,
                self.simplify,
            )
        else:
            return MultiCVRP(
                self.num_vehicles,
                self.trips,
                self.depot,
                self.distance_matrix,
                self.capacities,
                self.locations,
                self.use_deliveries,
                self.simplify,
            )
