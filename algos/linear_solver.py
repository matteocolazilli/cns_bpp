from ortools.linear_solver import pywraplp
from algos.solution import Solution
from typing import List, Dict, Any


def create_data_model(n: int, c: int, weights: List[int]) -> Dict[str, Any]:
    """Create the data for the example."""
    data = {}
    data["weights"] = weights
    data["items"] = list(range(n))
    data["bins"] = data["items"]
    data["bin_capacity"] = c
    return data



def linear_solver(n: int, c: int, weights: List[int]) -> Solution:
    data = create_data_model(n, c, weights)

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")

    if not solver:
        return

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    x = {}
    for i in data["items"]:
        for j in data["bins"]:
            x[(i, j)] = solver.IntVar(0, 1, "x_%i_%i" % (i, j))

    # y[j] = 1 if bin j is used.
    y = {}
    for j in data["bins"]:
        y[j] = solver.IntVar(0, 1, "y[%i]" % j)

    # Constraints
    # Each item must be in exactly one bin.
    for i in data["items"]:
        solver.Add(sum(x[i, j] for j in data["bins"]) == 1)

    # The amount packed in each bin cannot exceed its capacity.
    for j in data["bins"]:
        solver.Add(
            sum(x[(i, j)] * data["weights"][i] for i in data["items"])
            <= y[j] * data["bin_capacity"]
        )

    # Objective: minimize the number of bins used.
    solver.Minimize(solver.Sum([y[j] for j in data["bins"]]))

    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        num_bins = 0
        for j in data["bins"]:
            if y[j].solution_value() == 1:
                bin_items = []
                bin_weight = 0
                for i in data["items"]:
                    if x[i, j].solution_value() > 0:
                        bin_items.append(i)
                        bin_weight += data["weights"][i]
                if bin_items:
                    num_bins += 1
                    print("Bin number", j)
                    print("  Items packed:", bin_items)
                    print("  Total weight:", bin_weight)
                    print()
        print()
        print("Number of bins used:", num_bins)
        print("Time = ", solver.WallTime(), " milliseconds")

    elif status == pywraplp.Solver.INFEASIBLE:
        print("The problem is infeasible.")
    elif status == pywraplp.Solver.ABNORMAL:
        print("The problem is abnormal.")
    elif status == pywraplp.Solver.NOT_SOLVED:
        print("The problem is not solved.")
    elif status == pywraplp.Solver.FEASIBLE:
        print("The problem is feasible, but not optimal.")
    elif status == pywraplp.Solver.UNKNOWN:
        print("The problem is unknown.")
    elif status == pywraplp.Solver.OPTIMAL:
        print("The problem has an optimal solution.")
    elif status == pywraplp.Solver.INTERRUPTED:
        print("The problem was interrupted.")
    elif status == pywraplp.Solver.ERROR:
        print("The problem has an error.")
    elif status == pywraplp.Solver.TIME_LIMIT:
        print("The problem has a time limit.")
    elif status == pywraplp.Solver.SATISFIABLE:
        print("The problem is satisfiable.")
    elif status == pywraplp.Solver.UNSATISFIABLE:
        print("The problem is unsatisfiable.")
    elif status == pywraplp.Solver.UNBOUNDED:   
        print("The problem is unbounded.")
    
    return None