from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp

from src.model.VRP import VRP
from src.model.adapter.Adapter import Adapter


class QiskitAdapter(Adapter):
    """
    A class to represent an adapter for the CPLEX model.

    Attributes:
        model (Model): CPLEX model to be adapted.
    """

    def __init__(self, qubo: VRP):
        self.model = qubo.model
        super().__init__(qubo)

    def add_vars(self):
        """
        Add the QUBO variables to the CPLEX model.
        They are already present in the model imported from QuboVRP.
        """
        pass

    def add_objective(self):
        """
        Add the QUBO objective function to the CPLEX model.
        """

        self.model.minimize(self.qubo.objective)

    def add_constraints(self):
        """
        Add the QUBO constraints to the CPLEX model.
        """

        for constraint in self.qubo.constraints:
            self.model.add_constraint(constraint)

    def solver_model(self) -> QuadraticProgram:
        """
        Builds the quadratic program for CVRP, based on CPLEX.
        """

        qp = from_docplex_mp(self.model)
        if self.qubo.simplify:
            qp = qp.substitute_variables(self.qubo.get_simplified_variables())
        return qp
