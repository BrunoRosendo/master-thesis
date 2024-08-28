from dotenv import load_dotenv

from src.model.dispatcher import RPP, CVRP
from src.solver.ClassicSolver import ClassicSolver

if __name__ == "__main__":
    load_dotenv()

    model = CVRP(1, [(46, 32), (20, 32), (71, 32), (46, 60), (46, 4)], 5, [1] * 5)

    solver = ClassicSolver(model)
    solution = solver.solve()
    solution.display()
