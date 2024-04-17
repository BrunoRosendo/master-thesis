from dotenv import load_dotenv

from src.solver.qubo.CplexSolver import CplexSolver, get_backend_sampler

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
        sampler=get_backend_sampler(),
    )
    result = cvrp.solve()
    result.display()
