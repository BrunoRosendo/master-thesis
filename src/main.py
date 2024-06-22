from dotenv import load_dotenv
from dwave.system import DWaveSampler, LeapHybridCQMSampler

from src.model.VRPSolution import VRPSolution
from src.solver.ClassicSolver import ClassicSolver
from src.solver.qubo.DWaveSolver import DWaveSolver
from src.solver.qubo.QiskitSolver import QiskitSolver, get_backend_sampler

if __name__ == "__main__":
    cvrp = DWaveSolver(
        1,
        None,
        [
            (456, 320),
            (228, 0),
            (912, 0),
            # (0, 80),
            # (114, 80),
            # (570, 160),
            # (798, 160),
            # (342, 240),
            # (684, 240),
            # (570, 400),
            # (912, 400),
            # (114, 480),
            # (228, 480),
            # (342, 560),
            # (684, 560),
            # (0, 640),
            # (798, 640),
        ],
        [
            (0, 3, 5),
            (2, 1, 6),
        ],
        False,
        # sampler=LeapHybridCQMSampler(),
    )
    result = cvrp.solve()
    # result = VRPSolution.from_json("300+302-3v")
    result.display()
