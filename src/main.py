from dotenv import load_dotenv

from src.model.dispatcher import RPP, CVRP
from src.solver.ClassicSolver import ClassicSolver

if __name__ == "__main__":
    load_dotenv()

    model = CVRP(1, [(46, 32), (20, 32), (71, 32), (46, 60), (46, 4)], 5, [1] * 5)
    m2 = RPP(1, [(5, 5), (15, 10)], None, [])

    solver = ClassicSolver(m2)
    solution = solver.solve()
    solution.display()
