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

    def solve(self):
        """
        Solve the CVRP using QUBO implemented in Qiskit.
        """
        cplex = self.get_cplex_model()
        qp = from_docplex_mp(cplex)

        if self.classical_solver:
            result = self.solve_classic(qp)
        else:
            qubo = self.quadratic_to_qubo(qp)

            print(f"The number of variables is {qubo.get_num_vars()}")
            print(qubo.prettyprint())

            # Solve the QUBO problem
            # optimizer = MinimumEigenOptimizer(min_eigen_solver=self.min_eigen_solver)
            # result = optimizer.solve(qubo)

        # Get the solution
        # solution = self.convert_solution(result)
        # solution.display()

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
            model.add_constraint(model.sum(x[i, j] for j in range(num_locations)) == 1)
            model.add_constraint(model.sum(x[j, i] for j in range(num_locations)) == 1)

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
                    u[i - 1] - u[j - 1] + (capacity * x[i, j])
                    <= capacity - self.get_location_demand(j)
                )

        return model

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
        print(result)
        return result
