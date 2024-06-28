from dotenv import load_dotenv

from src.model.VRPSolution import VRPSolution

if __name__ == "__main__":
    load_dotenv()

    # cvrp = QiskitSolver(
    #     2,
    #     [12, 10],
    #     [
    #         (46, 32),
    #         (40, 0),
    #         (91, 0),
    #         (0, 8),
    #         (57, 40),
    #         (91, 40),
    #         (11, 48),
    #         (23, 48),
    #     ],
    #     [(1, 6, 5), (2, 7, 6), (4, 3, 4), (5, 0, 2), (5, 1, 3)],
    #     True,
    #     sampler=get_backend_sampler(),
    # )
    # result = cvrp.solve()
    # result.save_json("P-n16-k8-qaoa")
    result = VRPSolution.from_json("rpp-n16-k2-cqm")
    result.display()
