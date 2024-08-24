from dotenv import load_dotenv

from src.model.VRPSolution import VRPSolution
from src.model.cvrp.ConstantCVRP import ConstantCVRP
from src.model.cvrp.InfiniteCVRP import InfiniteCVRP
from src.model.cvrp.MultiCVRP import MultiCVRP
from src.model.rpp.CapacityRPP import CapacityRPP
from src.model.rpp.InfiniteRPP import InfiniteRPP
from src.solver.ClassicSolver import ClassicSolver

if __name__ == "__main__":
    load_dotenv()

    VRPSolution.from_json("18").display()
