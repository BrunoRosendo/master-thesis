from dotenv import load_dotenv

from src.solver.ClassicSolver import ClassicSolver
from src.solver.qubo.CplexSolver import CplexSolver

if __name__ == "__main__":
    load_dotenv()

    cvrp = CplexSolver(
        2,
        [1, 3],
        [
            (456, 320),
            (228, 0),
            (100, 100),
            (50, 50),
            (150, 0)
        ],
        [],
        False,
        True,
        simplify=True,
    )
    result = cvrp.solve()
    result.display()
