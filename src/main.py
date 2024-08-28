from dotenv import load_dotenv
from dwave.system import LeapHybridCQMSampler

from src.model.dispatcher import CVRP
from src.solver.qubo.DWaveSolver import DWaveSolver

if __name__ == "__main__":
    load_dotenv()

    model = CVRP(1, [(46, 32), (20, 32)], 5, [1] * 5)

    solver = DWaveSolver(model, sampler=LeapHybridCQMSampler())
    solution = solver.solve()
    solution.display()
