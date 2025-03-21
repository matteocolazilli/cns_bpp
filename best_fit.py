from typing import List


def best_fit(items: List[int], weights: List[int], c: int) -> List[List[int]]:
    """
    Algoritmo Best Fit per il Bin Packing:
    Per ogni oggetto, cerca il bin con il minimo spazio in cui può essere inserito.
    Se non c'è spazio in nessun bin, crea un nuovo bin.
    Args: 
        items (List[int]): Lista di indici degli oggetti da assegnare.
        weights (List[int]): Lista dei pesi degli oggetti.
        c (int): Capacità massima di ciascun bin.
    Returns:
        List[List[int]]: Lista di bin, ognuno con la lista di indici degli oggetti assegnati.
    """
    bins: List[List[int]] = []
    bin_space: List[int] = []  # Tiene traccia dello spazio rimanente in ogni bin

    for i in items:
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

    return bins
