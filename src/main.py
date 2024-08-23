from dotenv import load_dotenv

from src.model.cvrp.InfiniteCVRP import InfiniteCVRP
from src.solver.ClassicSolver import ClassicSolver

if __name__ == "__main__":
    load_dotenv()

    model = InfiniteCVRP(1, [(46, 32), (20, 32), (71, 32), (46, 60), (46, 4)], [1] * 5)
    cvrp = ClassicSolver(model)
    result = cvrp.solve()
    result.display()
