from dotenv import load_dotenv

from src.model.VRPSolution import VRPSolution

if __name__ == "__main__":
    load_dotenv()

    # cvrp = DWaveSolver(
    #     1,
    #     None,
    #     [
    #         (456, 320),
    #         (228, 0),
    #         (912, 0),
    #         (0, 80),
    #     ],
    #     [
    #         (2, 1, 6),
    #         (3, 0, 5),
    #     ],
    #     True,
    # )
    # result = cvrp.solve()
    # result.save_json("line test")

    result = VRPSolution.from_json("18")
    result.display(fig_height=1000)
