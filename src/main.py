from dotenv import load_dotenv
from dwave.system import DWaveSampler

from src.solver.qubo.DWaveSolver import DWaveSolver

if __name__ == "__main__":
    load_dotenv()

    cvrp = DWaveSolver(
        2,
        None,
        [(46, 32), (20, 32), (71, 32), (46, 60), (46, 4)],
        [],
        False,
        sampler=DWaveSampler(),
        num_reads=1000,
    )
    result = cvrp.solve()
    # result.save_json("P-n16-k8-qaoa")
    # result = VRPSolution.from_json("rpp-n16-k2-cqm")
    result.display()
