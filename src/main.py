from dotenv import load_dotenv

from src.model.VRPSolution import VRPSolution
from src.solver.qubo.CplexSolver import CplexSolver

if __name__ == "__main__":
    load_dotenv()

    cvrp = CplexSolver(
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
        classical_solver=True,
        # sampler=get_backend_sampler(),
    )
    # result = cvrp.solve()
    # result.save_json("test")
    result = VRPSolution.from_json("test")
    result.display()
