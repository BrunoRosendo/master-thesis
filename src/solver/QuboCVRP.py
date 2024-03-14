from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer, OptimizationResult
from qiskit_optimization.converters import (
    InequalityToEquality,
    IntegerToBinary,
    LinearEqualityToPenalty,
)

from src.model.CPLEXModel import CPLEXModel
from src.model.CVRPSolution import CVRPSolution
from src.model.DiffCapModel import DiffCapModel
from src.model.SameCapModel import SameCapModel
from src.solver.CVRP import CVRP


class QuboCVRP(CVRP):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with QUBO algorithm, using Qiskit.

    Attributes:
    - classical_solver (bool): Whether to use a classical solver to solve the QUBO problem.
    """

    def __init__(
        self,
        num_vehicles: int,
        capacities: int | list[int] | None,
        locations: list[tuple[int, int]],
        trips: list[tuple[int, int, int]],
        classical_solver=False,
    ):
        super().__init__(num_vehicles, capacities, locations, trips)
        self.classical_solver = classical_solver

    def _solve_cvrp(self) -> OptimizationResult:
        """
        Solve the CVRP using QUBO implemented in Qiskit.
        """
        qp = self.model.quadratic_program(False)

        if self.classical_solver:
            result = self.solve_classic(qp)
        else:
            qubo = self.quadratic_to_qubo(qp)

            print(f"The number of variables is {qubo.get_num_vars()}")
            print(qubo.prettyprint())

            # Solve the QUBO problem
            # optimizer = MinimumEigenOptimizer(min_eigen_solver=self.min_eigen_solver)
            # result = optimizer.solve(qubo)

        if result.status.name != "SUCCESS":
            raise Exception("Failed to solve the problem!")

        return result

    def quadratic_to_qubo(self, qp: QuadraticProgram) -> QuadraticProgram:
        """
        Convert the quadratic program to a QUBO problem, using the Qiskit converters.
        """

        # If there are inequality constraints
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
        optimizer = CplexOptimizer()
        result = optimizer.solve(qp)
        return result

    def _convert_solution(self, result: OptimizationResult) -> CVRPSolution:
        """
        Convert the optimizer result to a CVRPSolution solution.
        """
        var_dict = result.variables_dict
        route_starts = self.get_result_route_starts(var_dict)

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
                index = self.get_result_next_location(var_dict, index)

            routes.append(route)
            distances.append(route_distance)
            loads.append(route_loads)
            total_distance += route_distance

        return CVRPSolution(
            self.num_vehicles,
            self.locations,
            result.fval,
            total_distance,
            routes,
            distances,
            self.depot,
            loads if self.use_capacity else None,
        )

    def get_var_name(self, i: int, j: int) -> str:
        """
        Get the name of a variable.
        """
        return f"x_{i}_{j}"

    def get_result_route_starts(self, var_dict: dict[str, float]) -> list[int]:
        """
        Get the starting location for each route from the variable dictionary.
        """
        route_starts = []

        cur_location = 1
        while len(route_starts) < self.num_vehicles:
            var_value = var_dict[self.get_var_name(0, cur_location)]
            if var_value == 1.0:
                route_starts.append(cur_location)
            cur_location += 1

        return route_starts

    def get_result_next_location(
        self, var_dict: dict[str, float], cur_location: int
    ) -> int | None:
        """
        Get the next location for a route from the variable dictionary.
        """
        for i in range(len(self.locations)):
            var_value = var_dict[self.get_var_name(cur_location, i)]
            if var_value == 1.0:
                return i
        return None

    def get_model(self) -> CPLEXModel:
        """
        Get a cplex instance of the CVRPModel.
        """

        if self.same_capacity:
            return SameCapModel(
                self.num_vehicles,
                self.trips,
                self.depot,
                self.distance_matrix,
                self.capacities,
            )
        else:
            return DiffCapModel(
                self.num_vehicles,
                self.trips,
                self.depot,
                self.distance_matrix,
                self.capacities,
            )
