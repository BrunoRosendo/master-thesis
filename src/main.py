from dotenv import load_dotenv
from dwave.system import DWaveSampler, LeapHybridCQMSampler

from src.model.VRPSolution import VRPSolution
from src.solver.ClassicSolver import ClassicSolver
from src.solver.qubo.DWaveSolver import DWaveSolver
from src.solver.qubo.QiskitSolver import QiskitSolver, get_backend_sampler

if __name__ == "__main__":
    load_dotenv()
    result = VRPSolution.from_json("300+302-3v")
    result.display()
