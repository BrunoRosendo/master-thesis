from dotenv import load_dotenv

from src.model.VRPSolution import VRPSolution

if __name__ == "__main__":
    load_dotenv()

    # cvrp = ClassicSolver(
    #     1,
    #     None,
    #     [
    #         (41.165063341024, -8.55722805166063),
    #         (41.1570708790605, -8.51599248615023),
    #         (41.1910277341427, -8.5216944398509),
    #     ],
    #     [],
    #     False,
    #     distance_function=distance_api,
    #     distance_unit=DistanceUnit.METERS,
    # )
    # result = cvrp.solve()
    result = VRPSolution.from_json("300+302-2v-test")
    result.display()
