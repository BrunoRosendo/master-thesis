from dotenv import load_dotenv
from dwave.system import DWaveSampler

from src.solver.qubo.DWaveSolver import DWaveSolver

if __name__ == "__main__":
    load_dotenv()

    cvrp = DWaveSolver(
        1,
        None,
        [
            (456, 320),
            (228, 0),
        ],
        [
            (0, 1, 6),
        ],
        True,
        sampler=DWaveSampler(),
    )
    result = cvrp.solve()
    result.print()
