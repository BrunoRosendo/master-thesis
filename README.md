# Master's Thesis - Quantum Solver for Routing Problems

This repository contains the code for the Master's Thesis project "Quantum Algorithms for Optimizing Urban Transportation" developed by Bruno Rosendo. The thesis can be found at [Repositório Aberto](https://hdl.handle.net/10216/160532).

Feel free to use this code as a reference for your projects and experiments, but please keep the licence intact.

## How to Use

To use the solvers, you must choose your preferred model (CVRP or RPP) and solver (D-Wave, Qiskit or Classic) and set the parameters for the routing problem and platform. You can then solve and visualize or save the results.

An example of a simple CVRP problem with the D-Wave solver is shown below:

```python
from src.model.dispatcher import CVRP
from src.solver.qubo.DWaveSolver import DWaveSolver

# Define the problem parameters
model = CVRP(1, [(46, 32), (20, 32), (71, 32), (46, 60), (46, 4)], 5, [1] * 5)

# Define the solver
solver = DWaveSolver(model)

# Solve the problem
solution = solver.solve()

# Display the solution
solution.display()
```

### Choosing the Model

Currently, the project supports four routing problem variations: Vehicle Routing Problem (VRP), Capacitated VRP (CVRP), Multi-Capacitated VRP (MCVRP) and Ride Pooling Problem (RPP). You can find details about each problem in the [thesis document](https://repositorio-aberto.up.pt/handle/10216/160532).

The model be chosen by using one of two dispatcher functions: `CVRP` or `RPP`, depending on which problem you're trying to solve. The functions reside in `src/model/dispatcher.py`. The right variation will be used depending on the parameters you set.

These functions use a set of parameters that will define the routing problem at hand. These parameters are in the table:

| Parameter         | Type                       | Description                                                                                                                                      | Default            |
|-------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `num_vehicles`    | int                        | Number of vehicles in the problem.                                                                                                               | -                  |
| `capacities`      | int \| list[int] \| _None_ | Capacity for the vehicles. You can also specify for each or use _None_ for infinite capacity.                                                    | -                  |
| `locations`       | list[tuple[float, float]]  | List of coordinates representing the locations in the problem.                                                                                   | -                  |
| `demands`         | list[int]                  | List of demands for each location. Must be specified if capacities is not _None_. The minimum demand is 1 for each location. Used only for CVRP. | _None_             |
| `trips`           | list[tuple[int, int, int]] | List of trips in the format _(src, dest, demand)_. Used only for RPP.                                                                            | -                  |
| `cost_function`   | Callable                   | Cost function used to generate a distance matrix. See [below](#cost-functions) for details.                                                      | Manhattan Distance |
| `distance_matrix` | list[list[float]]          | Optional parameter to set the distance matrix directly.                                                                                          | _None_             |
| `location_names`  | list[str]                  | Optional list of location names used for display.                                                                                                | _None_             |
| `distance_unit`   | _DistanceUnit_             | Unit used for distance/cost. Used for proper visualization.                                                                                      | Meters             |

### Choosing the Solver

#### D-Wave Solver (Leap)

The `DWaveSolver` interacts with [D-Wave's Leap](https://cloud.dwavesys.com/leap/) cloud provider and is this project's most recommended quantum solver.

To set it up, you need to authenticate yourself in one of three ways:
- Copy your API token from the platform and create a `.env` file in the project's root with the line `DWAVE_API_TOKEN=<token>`.
- Create a `DWAVE_API_TOKEN` environment variable under your Python (virtual) environment with your API token.
- Install the [dwave-ocean-sdk](https://docs.ocean.dwavesys.com/en/latest/docs_cli.html) package and authenticate there.

This solver has the following parameters that can be configured:

| Parameter           | Type                                                                                          | Description                                                                                                                                                                                                                                                             | Default                                                                                                                                          |
|---------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `track_progress`    | bool                                                                                          | Whether to track progress                                                                                                                                                                                            by printing logs every iteration of the algorithm. | True                                                                                                                                             |                                                                                                                                                                                                                      |         |
| `sampler`           | [Sampler](https://docs.ocean.dwavesys.com/en/stable/overview/stack.html#bottom-up-approach)   | The sampler for the annealing. It can be a classical or quantum sampler.                                                                                                                                                                                                | [ExactCQMSolver](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampler_composites/samplers.html#exact-cqm-solver)               |
| `embedding`         | [Embedding](https://docs.ocean.dwavesys.com/en/stable/overview/stack.html#bottom-up-approach) | Embedding class used to embed the problem into the quantum hardware.                                                                                                                                                                                                    | [EmbeddingComposite](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/composites.html#dwave.system.composites.EmbeddingComposite) |
| `embed_bqm`         | bool                                                                                          | If using a BQM sampler, locally embed the problem before sampling.                                                                                                                                                                                                      | True                                                                                                                                             |
| `num_reads`         | int                                                                                           | If using a BQM sampler, sets a fixed number of reads.                                                                                                                                                                                                                   | _None_                                                                                                                                           |
| `time_limit`        | int                                                                                           | Optionally sets a time limit for the annealing process in seconds. If not specified, the sampler usually has its default value.                                                                                                                                         | _None_                                                                                                                                           |
| `embedding_timeout` | int                                                                                           | Optionally sets a time limit for the embedding process in seconds. If not specified, the sampler usually has the default value of 1000 seconds.                                                                                                                         | _None_                                                                                                                                           |

#### Qiskit Solver (IBM)

The `QiskitSolver` interacts with the [IBM Quantum](https://quantum.ibm.com/) cloud platform.

To set it up, you need to authenticate yourself in one of two ways:
- Copy your API token from the platform and create a `.env` file in the project's root with the line `IBM_TOKEN=<token>`.
- Create an `IBM_TOKEN` environment variable under your Python (virtual) environment with your API token.

This solver has the following parameters that can be configured:

| Parameter             | Type                                                                                                                                            | Description                                                                                                                                                                                                                                                             | Default                                                                                                                           |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `track_progress`      | bool                                                                                                                                            | Whether to track progress                                                                                                                                                                                            by printing logs every iteration of the algorithm. | True                                                                                                                              |                                                                                                                                                                                                                      |         |
| `classical_solver`    | bool                                                                                                                                            | Whether to use a classical optimizer (CPLEX) instead of a quantum optimizer.                                                                                                                                                                                            | False                                                                                                                             |
| `sampler`             | [Sampler](https://docs.quantum.ibm.com/api/qiskit/qiskit.primitives.Sampler)                                                                    | The sampler used in the quantum optimizer. It can either be a simulator or a real quantum computer.                                                                                                                                                                     | [Sampler](https://docs.quantum.ibm.com/api/qiskit/qiskit.primitives.Sampler)                                                      |
| `classical_optimizer` | [Optimizer](https://docs.quantum.ibm.com/api/qiskit/0.28/qiskit.algorithms.optimizers.Optimizer)                                                | Classical optimizer used to decide new parameters in between QAOA iterations.                                                                                                                                                                                           | [COBYLA](https://docs.quantum.ibm.com/api/qiskit/0.28/qiskit.algorithms.optimizers.COBYLA)                                        |
| `warm_start`          | bool                                                                                                                                            | Whether to run QAOA with a warm start.                                                                                                                                                                                                                                  | False                                                                                                                             |
| `pre_solver`          | [OptimizationAlgorithm](https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.OptimizationAlgorithm.html) | Classical optimizer used to pre-solve the problem if warm start is used.                                                                                                                                                                                                | [CplexOptimizer](https://qiskit-community.github.io/qiskit-optimization/stubs/qiskit_optimization.algorithms.CplexOptimizer.html) |

#### Classical Solver (OR-Tools)

The `ClassicalSolver` uses the [Google OR-Tools](https://developers.google.com/optimization) library to solve the routing problem classically. This is a great way to test your inputs or compare the quantum solvers with a well-known classical solver.

This solver does not require any authentication, as it is a local solver. It has the following parameters that can be configured:

| Parameter                               | Type                                                                                                               | Description                                                                                                                                                                                                                                                             | Default   |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| `track_progress`                        | bool                                                                                                               | Whether to track progress                                                                                                                                                                                            by printing logs every iteration of the algorithm. | True      |                                                                                                                                                                                                                      |         |
| `solution_strategy`                     | [FirstSolution](https://developers.google.com/optimization/routing/routing_options#first_solution_strategy) (enum) | First solution strategy. The method used to find an initial solution.                                                                                                                                                                                                   | Automatic |
| `local_search_metaheuristic`            | [LocalSearch](https://developers.google.com/optimization/routing/routing_options#local_search_options) (enum)      | Local search strategy (metaheuristic) used by the solver.                                                                                                                                                                                                               | Automatic |
| `distance_global_span_cost_coefficient` | int                                                                                                                | The coefficient is multiplied by each vehicle’s travelled distance and is helpful to distribute distance between routes.                                                                                                                                                | 1         |
| `time_limit_seconds`                    | int                                                                                                                | Maximum execution time in seconds.                                                                                                                                                                                                                                      | 10        |
| `max_distance_capacity`                 | int                                                                                                                | Maximum capacity for the distance dimension within OR-Tools. This doesn't need to be changed unless some overflow error happens.                                                                                                                                        | 90000     |

#### Cost functions

A cost function generates a distance matrix from the locations list. The default cost function is the Manhattan distance, but you can define your own cost function by creating a function that takes a list of coordinates and returns the matrix. For example:

```python
from src.model.VRP import DistanceUnit

def manhattan_distance(
    locations: list[tuple[int, int]], unit: DistanceUnit = DistanceUnit.METERS
) -> list[list[float]]:
    """
    Compute the Manhattan distance between all locations.
    """

    return [
        [
            abs(from_location[0] - to_location[0])
            + abs(from_location[1] - to_location[1])
            for to_location in locations
        ]
        for from_location in locations
    ]
  ``` 

The project currently supports the following distance units: `manhattan_distance`, `euclidean_distance`, `haversine_distance` and `distance_api`.

The last one, `distance_api`, is a particular cost function that uses Google's [Distance Matrix API](https://developers.google.com/maps/documentation/distance-matrix/overview) to calculate the distance between locations. This is useful for real-world problems where the distance matrix is unknown beforehand. To use it, you must set the `GOOGLE_API_KEY` environment variable with your API key by adding it to the `.env` file or setting it in your (virtual) environment.

### Running the Solver

After defining the problem and choosing the solver, you can run the solver with the `solve()` method. This will return a `VRPSolution` object. You can then visualize the solution in the browser with the `display()` method, print it to the console using the `print()` method, or save it to a file with the `save_json()` method.

Loading a solution from a JSON file using the VRPSolution `from_json()` static method is also possible. This is useful for comparing solutions or visualizing them later.

## Development setup

If you want to contribute to this project, follow the instructions below to set up your development environment. Feel free to open issues or pull requests with questions or suggestions.

### Prerequisites

- [Python 3.10+](https://www.python.org/)
- [pip3](https://pypi.org/project/pip/)
- All the dependencies listed in the `requirements.txt` file, installed via pip

### Running

The entry point for the application is usually the `src/main.py` file. You can also use other scripts under `src/scripts/` or create your own. To run them, execute Python with the desired script.

### Code Formatting

We use the [Black](https://pypi.org/project/black/) formatter for all Python code. There is no specific linter, as Black is an opinionated formatter that enforces a consistent style.

## Project Structure

- `data/` - Data files used in the project.
  - `inputs/` - Benchmark inputs from Best-Known Solutions datasets.
  - `Porto/` - Data extracted from Porto's public transportation system.
- `results/` - Results from the VRP solvers, including JSON files and HTML pages.
- `src/` - Main source code directory.
  - `model/` - All classes and functions related to the VRP models and adapters.
  - `qiskit_algorithms/` - Modified version of the [qiskit-algorithms](https://github.com/qiskit-community/qiskit-algorithms) used in the project.
  - `scripts/` - Scripts for running experiments, generating plots, etc.
  - `solvers/` - Quantum and classical solvers for the VRP.