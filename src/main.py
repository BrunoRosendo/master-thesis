from dotenv import load_dotenv

from src.model.cvrp.ConstantCVRP import ConstantCVRP
from src.model.cvrp.InfiniteCVRP import InfiniteCVRP
from src.model.cvrp.MultiCVRP import MultiCVRP
from src.model.rpp.CapacityRPP import CapacityRPP
from src.model.rpp.InfiniteRPP import InfiniteRPP
from src.solver.ClassicSolver import ClassicSolver

if __name__ == "__main__":
    load_dotenv()

    model = CapacityRPP(
        1, [(46, 32), (20, 32), (71, 32), (46, 60), (46, 4)], [(0, 1, 2)], [2]
    )
    cvrp = ClassicSolver(model)
    result = cvrp.solve()
    result.display()
