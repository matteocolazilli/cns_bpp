import logging
import sys
import time
from typing import List, Tuple
import os
def log_function_call(func):
    def wrapper(*args, **kwargs):
        
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
        logging.info(f"---->Calling {func.__name__} :\n")
        logging.info(f"Bins: {len(args[0].bins)}")
        logging.info(f"BINS:\n{args[0].bins}\nTRASH:\n{args[0].trash}\n")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        result_sol = result[0]
        logging.info(f"Bins: {len(result_sol.bins)}")
        logging.info(f"{func.__name__} executed for {end-start} seconds \nReturned this:\n{result_sol.bins}\nTRASH:\n{result_sol.trash}\nOBJ: {result_sol.objective_value()}\nComplete: {result[1]}\n")
        
        return result
    return wrapper

def read_instance(file_path: str) -> Tuple[int, int, List[int]]:
    """
    Legge un file di istanza nel formato BPP:
      - Prima riga: numero di oggetti (n)
      - Seconda riga: capacità dei bin (c)
      - Dalle righe successive: per ciascun oggetto j (j=1,...,n) il peso wj
    Args:
      file_path (str): Percorso del file di istanza.
    Returns:
      (items, weights, c)
    """
    with open(file_path, "r",encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 2:
        raise ValueError("Formato del file non valido: attese almeno 2 righe (n e c).")
    try:
        n = int(lines[0])
    except ValueError as exc:
        raise ValueError("La prima riga deve essere un intero (numero di oggetti).") from exc
    try:
        c = int(lines[1])
    except ValueError as exc:
        raise ValueError("La seconda riga deve essere un intero (capacità dei bin).") from exc
    if len(lines) < 2 + n:
        raise ValueError(f"Formato del file non valido: attesi {n} pesi, trovati {len(lines)-2}.") from exc
    weights: List[int] = []
    for i in range(n):
        try:
            weight = int(lines[2 + i])
        except ValueError as exc:
            raise ValueError(f"Impossibile convertire il peso (riga {2+i+1}).") from exc
        weights.append(weight)
    return n, c, weights



def sorted_dir_list(directory):
    with os.scandir(directory) as entries:
        sorted_entries = sorted(entries, key=lambda entry: entry.name)
        sorted_items = [entry.name for entry in sorted_entries]
    return sorted_items