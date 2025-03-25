from typing import List
from .solution import Solution
from .auxiliary import random_instance
import random

def best_fit(n: int, c: int, weights: List[int]) -> Solution:
    """
    Algoritmo Best Fit per il Bin Packing:
    Per ogni oggetto, cerca il bin con il minimo spazio in cui può essere inserito.
    Se non c'è spazio in nessun bin, crea un nuovo bin.
    Args: 
        n (int): Numero di oggetti.
        weights (List[int]): Lista dei pesi degli oggetti.
        c (int): Capacità massima di ciascun bin.
    Returns:
        List[List[int]]: Lista di bin, ognuno con la lista di indici degli oggetti assegnati.
    """
    bins: List[List[int]] = []
    bin_space: List[int] = []  # Tiene traccia dello spazio rimanente in ogni bin

    for i in range(n):
        min_idx = -1
        min_space = c + 1 

        # Trova il bin con il minor spazio sufficiente per l'oggetto
        for j in range(len(bins)):
            if weights[i] <= bin_space[j] < min_space:
                min_space = bin_space[j]
                min_idx = j

        if min_idx == -1:
            # Crea un nuovo bin
            bins.append([i])
            bin_space.append(c - weights[i])
        else:
            # Inserisce l'oggetto nel bin migliore trovato
            bins[min_idx].append(i)
            bin_space[min_idx] -= weights[i]

    return Solution(bins, [], c, weights)


if __name__ == "__main__":
    # Esempio di utilizzo dell'algoritmo Best Fit
    n, c, seed = 10, 20, random.randint(0, 1000)
    n, c, weights = random_instance(n, c, seed)
    print("Esempio di utilizzo dell'algoritmo Best Fit:")
    print(f"Numero di oggetti: {n}")
    print(f"Capacità massima dei bin: {c}")
    print(f"Pesi degli oggetti: {weights}")
    solution = best_fit(n, c, weights)
    solution.validate_assignment()
    print(f"Soluzione trovata:")
    for i, bin in enumerate(solution.bins):
        print(f"Bin {i+1}: {bin}")
    
    print(f"Numero di bin usati: {len(solution.bins)}")
    print(f"Seed utilizzato: {seed}")