from dimod import ConstrainedQuadraticModel
from dimod.sym import Sense
from docplex.mp.basic import Expr
from docplex.mp.linear import LinearExpr, ConstantExpr
from docplex.mp.quad import QuadExpr

from src.model.adapter.Adapter import Adapter


class DWaveAdapter(Adapter):
    """
    A class to represent an adapter for the DWave model.

    Attributes:
        model (ConstrainedQuadraticModel): DWave model to be adapted.
        copy_vars (bool): Flag to indicate if the variables should be copied when adding constraints.
    """

    def __init__(self, qubo):
        self.model = ConstrainedQuadraticModel()
        self.copy_vars = False
        super().__init__(qubo)

    def add_vars(self):
        """
        Add the QUBO variables to the DWave model.
        """

        for v in self.qubo.x:
            self.model.add_variable("BINARY", v.name)

        for v in self.qubo.u:
            self.model.add_variable(
                "INTEGER", v.name, lower_bound=v.lb, upper_bound=v.ub
            )

    def add_objective(self):
        """
        Add the QUBO objective function to the DWave model.
        """

        self.model.set_objective(self.expression_to_tuples(self.qubo.objective))

    def add_constraints(self):
        """
        Add the QUBO constraints to the DWave model.
        """

        for constraint in self.qubo.constraints:
            self.model.add_constraint(
                self.expression_to_tuples(constraint.left_expr),
                constraint.sense.operator_symbol,
                self.expression_to_tuples(constraint.right_expr),
                copy=self.copy_vars,
            )

    def model(self) -> ConstrainedQuadraticModel:
        """
        Returns the constrained quadratic model for VRP, based on DWave Ocean.
        Removes unnecessary constraints if simplify is set to True.
        """

        if self.qubo.simplify:
            self.model.fix_variables(self.qubo.get_simplified_variables())

            # Check redundant constraint (always true). Other senses don't implement __bool__()
            for key in list(self.model.constraints):
                constraint = self.model.constraints[key]
                if constraint.sense is Sense.Eq and constraint:
                    self.model.remove_constraint(key)

        return self.model

    def expression_to_tuples(self, expr: Expr) -> list[tuple]:
        """
        Convert an expression to a list of tuples, depending on the expression type.
        """

        if isinstance(expr, LinearExpr):
            return self._linear_expression_to_tuples(expr)
        elif isinstance(expr, QuadExpr):
            return self._quadratic_expression_to_tuples(expr)
        elif isinstance(expr, ConstantExpr):
            return expr.constant
        else:
            raise ValueError("Invalid expression type.")

    def _linear_expression_to_tuples(self, expr: LinearExpr) -> list[tuple[str, int]]:
        """
        Convert a linear expression to a list of tuples.
        The first element of the tuple is the variable name and the second is the coefficient.
        """

        return [(v.name, coefficient) for (v, coefficient) in expr.iter_terms()]

    def _quadratic_expression_to_tuples(self, expr: QuadExpr) -> list[tuple]:
        """
        Convert a quadratic expression to a list of tuples, including the linear part.
        The first two elements of the tuple are the variable names and the third is the coefficient.
        """

        quad_part = [
            (pair.first.name, pair.second.name, coefficient)
            for (pair, coefficient) in expr.iter_quads()
        ]
        linear_part = self._linear_expression_to_tuples(expr.linear_part)

        return quad_part + linear_part
