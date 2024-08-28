from dimod import ConstrainedQuadraticModel
from dimod.sym import Sense
from docplex.mp.basic import Expr
from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr, ConstantExpr
from docplex.mp.quad import QuadExpr

from src.model.VRP import VRP
from src.model.adapter.Adapter import Adapter


class DWaveAdapter(Adapter):
    """
    A class to represent an adapter for the DWave model.

    Attributes:
        model (ConstrainedQuadraticModel): DWave model to be adapted.
        copy_vars (bool): Flag to indicate if the variables should be copied when adding constraints.
        use_bqm (bool): Flag to indicate if the model should be converted to a BQM.
    """

    def __init__(self, qubo: VRP, use_bqm: bool):
        self.model = ConstrainedQuadraticModel()
        self.copy_vars = False
        self.use_bqm = use_bqm
        super().__init__(qubo)

    def add_vars(self):
        """
        Add the QUBO variables to the DWave model.
        """

        for v in self.qubo.x:
            self.model.add_variable("BINARY", v.name)

        for v in self.qubo.u:
            if self.use_bqm:
                self.add_bqm_int_var(v)
            else:
                self.model.add_variable(
                    "INTEGER", v.name, lower_bound=v.lb, upper_bound=v.ub
                )

    def add_bqm_int_var(self, v: Var):
        """
        Add an integer variable to the BQM. It differs from CQM because all integer variables
        must have a lower bound equal to 0. The strategy is to instead add a constraint.
        """

        self.model.add_variable("INTEGER", v.name, lower_bound=0, upper_bound=v.ub)

        if v.lb > 0:
            self.model.add_constraint([(v.name, 1)], Sense.Ge, v.lb)
        else:
            raise ValueError(
                "For BQM, integers' lower bound must be greater or equal to 0."
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
            )

    def solver_model(self) -> ConstrainedQuadraticModel:
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

    @staticmethod
    def _linear_expression_to_tuples(expr: LinearExpr) -> list[tuple[str, int]]:
        """
        Convert a linear expression to a list of tuples.
        The first element of the tuple is the variable name and the second is the coefficient.
        """

        return [(v.name, coefficient) for (v, coefficient) in expr.iter_terms()]

    @staticmethod
    def _quadratic_expression_to_tuples(expr: QuadExpr) -> list[tuple]:
        """
        Convert a quadratic expression to a list of tuples, including the linear part.
        The first two elements of the tuple are the variable names and the third is the coefficient.
        """

        quad_part = [
            (pair.first.name, pair.second.name, coefficient)
            for (pair, coefficient) in expr.iter_quads()
        ]
        linear_part = DWaveAdapter._linear_expression_to_tuples(expr.linear_part)

        return quad_part + linear_part
