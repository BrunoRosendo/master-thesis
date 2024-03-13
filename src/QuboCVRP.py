from CVRP import CVRP
from CVRPSolution import CVRPSolution
from docplex.mp.model import Model
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import (
    InequalityToEquality,
    IntegerToBinary,
    LinearEqualityToPenalty,
)
from qiskit_optimization.algorithms import CplexOptimizer


class QuboCVRP(CVRP):
    """
    Class for solving the Capacitated Vehicle Routing Problem (CVRP) with QUBO algorithm, using Qiskit.

    Attributes:
    - classical_solver (bool): Whether to use a classical solver to solve the QUBO problem.
    """

    def __init__(self, vehicles, depot, locations, trips, classical_solver=False):
        super().__init__(vehicles, depot, locations, trips)
        self.classical_solver = classical_solver

    def _solve_cvrp(self):
        """
        Solve the CVRP using QUBO implemented in Qiskit.
        """
        cplex = self.get_cplex_model()
        qp = from_docplex_mp(cplex)
        qp = self.simplify_problem(qp)

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

    def get_cplex_model(self):
        """
        Get the CPLEX model for the CVRP.
        """

        model = Model("CVRP")
        num_locations = len(self.distance_matrix)

        # Create variables
        x = model.binary_var_matrix(num_locations, num_locations, name="x")
        u = model.binary_var_list(range(1, num_locations), name="u")

        # Objective function
        objective = model.sum(
            self.distance_matrix[i][j] * x[i, j]
            for i in range(num_locations)
            for j in range(num_locations)
        )

        model.minimize(objective)

        # Constraints

        # Each location must be visited exactly once
        for i in range(1, num_locations):
            model.add_constraint(
                model.sum(x[i, j] for j in range(num_locations) if i != j) == 1
            )
            model.add_constraint(
                model.sum(x[j, i] for j in range(num_locations) if i != j) == 1
            )

        # All vehicles need to leave and return to the depot
        model.add_constraint(
            model.sum(x[0, i] for i in range(1, num_locations)) == self.num_vehicles
        )
        model.add_constraint(
            model.sum(x[i, 0] for i in range(1, num_locations)) == self.num_vehicles
        )

        # TODO: handle multiple capacities and setting for constant capacity
        capacity = self.vehicle_capacities[0]

        # Subtour elimination (MTV)
        for i in range(1, num_locations):
            for j in range(1, num_locations):
                if i == j:
                    continue

                model.add_constraint(
                    u[i - 1] - u[j - 1] + capacity * x[i, j]
                    <= capacity - self.get_location_demand(j)
                )

            model.add_constraint(u[i - 1] >= self.get_location_demand(i))
            model.add_constraint(u[i - 1] <= capacity)

        return model

    def simplify_problem(self, qp):
        """
        Simplify the problem by removing unnecessary variables.
        """

        for i in range(len(self.locations)):
            qp = qp.substitute_variables({f"x_{i}_{i}": 0})

        return qp

    def quadratic_to_qubo(self, qp):
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

    def solve_classic(self, qp):
        optimizer = CplexOptimizer()
        result = optimizer.solve(qp)
        return result

    def _convert_solution(self, result):
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
                cur_load += self.get_location_demand(index)
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

    def get_var_name(self, i, j):
        """
        Get the name of a variable.
        """
        return f"x_{i}_{j}"

    def get_result_route_starts(self, var_dict):
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

    def get_result_next_location(self, var_dict, cur_location):
        """
        Get the next location for a route from the variable dictionary.
        """
        for i in range(len(self.locations)):
            var_value = var_dict[self.get_var_name(cur_location, i)]
            if var_value == 1.0:
                return i
        return None
