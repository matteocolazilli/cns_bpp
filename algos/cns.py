from time import perf_counter, time
import random
import math
import os
import csv
import pandas as pd
import argparse
from itertools import combinations
from typing import List, Set, Tuple, Dict, Any
from algos.solution import Solution
from algos.auxiliary import log_function_call, read_instance, sorted_dir_list
from tqdm import tqdm

TABU_TIME_LIMIT = 1


    

#############################################
# Algoritmo First Fit per il Bin Packing
#############################################
def first_fit(items: List[int], weights: List[int], c: int) -> List[List[int]]:
    """
    Algoritmo First Fit per il Bin Packing:
    Per ogni oggetto, cerca il primo bin in cui può essere inserito.
    Se non c'è spazio in nessun bin, crea un nuovo bin.
    Args: 
        items (List[int]): Lista di indici degli oggetti da assegnare.
        weights (List[int]): Lista dei pesi degli oggetti.
        c (int): Capacità massima di ciascun bin.
    Returns:
        List[List[int]]: Lista di bin, ognuno con la lista di indici degli oggetti assegnati.
    """
    bins: List[List[int]] = []
    for i in items:
        placed = False
        for b in bins:
            if sum(weights[j] for j in b) + weights[i] <= c:
                b.append(i)
                placed = True
                break
        if not placed:
            bins.append([i])

    return bins


#############################################
# Procedura di riduzione preliminare
#############################################



def reduction(n: int, c: int, weights: List[int]) -> Tuple[List[int], List[List[int]]]:
    
    """
    Applica la procedura di riduzione: 
      Per ogni coppia di oggetti (i,j) tale che weights[i] + weights[j] == c,
      fissa tali oggetti in un nuovo bin e li rimuove dalla lista degli oggetti.
    Questo procedimento riduce il numero di oggetti da considerare nella ricerca.
    Args:
        n (int): Numero totale di oggetti.
        c (int): Capacità del bin.
        weights (List[int]): Lista dei pesi degli oggetti.

    Returns:
        Tuple[List[int], List[List[int]]]: Una tupla contenente la lista degli oggetti rimanenti
        e la lista dei bin fissati con le coppie di oggetti.
    """
    fixed_bins: List[List[int]] = []
    remaining_items = list(range(n))
    used = set()
    for i, j in combinations(remaining_items, 2):
        if i in used or j in used:
            continue
        if weights[i] + weights[j] == c:
            fixed_bins.append([i, j])
            used.update([i, j])
    remaining_items = [i for i in remaining_items if i not in used]

    return remaining_items, fixed_bins

#############################################
# Funzione pack_enumeration (pack_set)
#############################################

def pack_set(items: List[int], capacity: int, weights: List[int]) -> List[int]:
    """
    Implementa la procedura pack_set (Algorithm 3 del paper):
    Trova il sottoinsieme ottimale di 'items' (lista di indici) che massimizza il peso
    senza superare la capacity. In caso di parità, sceglie il sottoinsieme con meno oggetti.
    Args:
        items (List[int]): Lista di indici degli oggetti da considerare.
        capacity (int): Capacità massima del bin.
        weights (List[int]): Lista dei pesi degli oggetti.
    Returns:
        List[int]: Lista di indici degli oggetti da includere nel bin.
    """
    best_pack: List[int] = []
    best_weight = 0
    best_count = float('inf')

    # Funzione ricorsiva per la ricerca esaustiva
    def recurse(idx: int, current_pack: List[int], current_weight: int) -> None:
        nonlocal best_pack, best_weight, best_count
        # Caso base: capacità massima raggiunta
        if current_weight == capacity:
            if len(current_pack) < best_count:
                best_pack = current_pack[:]
                best_weight = current_weight
                best_count = len(current_pack)
            return
        # Caso base: capacità massima superata
        if current_weight > capacity:
            return

        # Caso base: tutti gli oggetti esaminati
        if idx == len(items):
            if (current_weight > best_weight) or (current_weight == best_weight and len(current_pack) < best_count):
                best_pack = current_pack[:]
                best_weight = current_weight
                best_count = len(current_pack)
            return

        # Includere items[idx] se possibile
        item = items[idx]
        if current_weight + weights[item] <= capacity:
            current_pack.append(item)
            recurse(idx + 1, current_pack, current_weight + weights[item])
            current_pack.pop()
        # Non includere items[idx]
        recurse(idx + 1, current_pack, current_weight)
    
    recurse(0, [], 0)
    return best_pack


#############################################
# Funzioni di ricerca locale: Descent e Tabu Search
#############################################

def swap(solution: Solution, b_index: int , S: set, T: set):
    """
    Esegue una mossa Swap sul bin indicato da b_index:
      - Rimuove gli oggetti in S dal bin b_index e li sposta nel trash.
      - Rimuove gli oggetti in T dal trash e li aggiunge al bin b_index.
    
    Args:
        solution (Solution): Soluzione corrente.
        b_index (int): Indice del bin su cui eseguire la mossa.
        S (set): Insieme di indici degli oggetti da spostare nel trash.
        T (set): Insieme di indici degli oggetti da spostare nel bin.
    """    
    
    for item in S:
        solution.bins[b_index].remove(item)
        solution.trash.add(item)
    for item in T:
        if item in solution.trash:
            solution.trash.remove(item)
        solution.bins[b_index].append(item)


def pack(solution: Solution, b_index: int):
    """
    Esegue la procedura Pack(b_index) sul bin indicato:
      - Unisce il bin con il trash e calcola il packing ottimale.
      - Aggiorna il bin con il nuovo packing e il trash con gli oggetti non inclusi.
    Args:
        solution (Solution): Soluzione corrente.
        b_index (int): Indice del bin su cui eseguire la procedura.
    """

    old_bin = set(solution.bins[b_index]) 
    old_trash = solution.trash.copy()
    full_union = old_bin.union(old_trash)
    
    # Se l'unione completa è di dimensione <= 20, esegui il packing completo
    if len(full_union) <= 20:
        union_items = list(full_union)
        new_pack = pack_set(union_items, solution.capacity, solution.weights)
        # Aggiorna il bin con il packing calcolato dall'unione completa
        solution.bins[b_index] = new_pack
        solution.trash = old_trash.union(old_bin) - set(new_pack)
    # Altrimenti, campiona un sottoinsieme di 10 oggetti da bin e trash
    else:
        # Campiona dal bin e dal trash
        sample_bin = random.sample(solution.bins[b_index], min(10, len(solution.bins[b_index])))
        sample_trash = random.sample(list(solution.trash), min(10, len(solution.trash)))
        union_sample = list(set(sample_bin + sample_trash))
        new_sample_pack = pack_set(union_sample, solution.capacity, solution.weights)
        
        # Preserva gli oggetti non campionati dal bin
        unsampled = old_bin - set(sample_bin)
        candidate_bin = new_sample_pack + list(unsampled)
        solution.bins[b_index] = candidate_bin
        # Aggiorna il trash in modo da preservare tutti gli oggetti originari:
        solution.trash = old_trash.union(old_bin) - set(candidate_bin) 

#@log_function_call
def descent(sol: Solution, attempts) -> Tuple[Solution, bool, int]:
    """
    Procedura di Descent:
      - Per ogni bin, esegue la procedura Pack(b) e verifica se c'è miglioramento.
        - Se c'è miglioramento, aggiorna la soluzione corrente.
        - Se la soluzione è completa, esegue il ripacchettamento degli oggetti nel trash.
        - Termina quando non ci sono più miglioramenti o quando la soluzione è completa.
    Args:
        sol (Solution): Soluzione corrente.
        attempts (int): Numero di tentativi di ripacchettamento.
    Returns:
        Tuple[Solution, bool, int]: La soluzione finale, un flag che indica se è completa e il numero di tentativi.
    """

    improvement_made = True
    complete_solution_found = False
    packing_attempts = attempts    
    
    # Ciclo principale di Descent
    while improvement_made and not complete_solution_found:
        
        improvement_made = False
        bin_indices = list(range(len(sol.bins)))
        random.shuffle(bin_indices)

        for b_idx in bin_indices:
            new_sol = sol.copy()
            pack(new_sol, b_idx)  # Esegue Pack(b_idx) su new_sol

            if not new_sol.check_trash_big_items() or not new_sol.check_bins_capacity():
                continue

            # Se c'è miglioramento nell'obiettivo, aggiorna la soluzione corrente
            if new_sol.objective_value() < sol.objective_value():
                sol = new_sol.copy()  # Aggiorna la soluzione corrente
                improvement_made = True
                
                # Se la soluzione è completa, esegui il ripacchettamento degli oggetti nel trash
                if sol.total_trash_weight() <= 2 * sol.capacity:
                    packing_attempts += 1
                    trash_items = list(sol.trash)
                    first_bin = pack_set(trash_items, sol.capacity, sol.weights)
                                
                    if sum(sol.weights[i] for i in first_bin) >= sol.total_trash_weight() - sol.capacity:
                        repacked = set(first_bin)
                        leftover = sol.trash - repacked
                        if len(leftover) > 0:
                            second_bin = list(leftover)
                            sol.bins.append(first_bin)
                            sol.bins.append(second_bin)
                        else: 
                            sol.bins.append(first_bin)
                        
                        sol.trash = set() 
                        complete_solution_found = True
                        break
                
    return sol, complete_solution_found, packing_attempts


#@log_function_call
def tabu_search(init_sol: Solution, time_limit: float, attempts: int, variant: bool) -> Tuple[Solution, bool, int]:
    """
    Procedura di Tabu Search completa:
      - Il limite massimo di iterazioni è definito come |B| * |I|.
      - Esplora mosse Swap(p, q) tra un bin e il trash, considerando l'intero vicinato
        (definito da PQ = {(0,1), (1,1), (2,1), (1,2), (2,2), (1,3), (3,1), (2,3), (3,2)}).
      - Per ogni mossa, se un oggetto (di peso w) viene spostato in un bin b, la tabu tenure
        per quella mossa è impostata come freq(b, w) / 2 (arrotondato per eccesso) + 1.
      - Se viene trovato un nuovo best, le frequenze vengono resettate.
      - Viene inoltre eseguita la variante che limita le mosse che degradano il secondo obiettivo.
    Args:
        init_sol (Solution): Soluzione iniziale.
        time_limit (float): Limite di tempo per la tabu search.
        attempts (int): Numero di tentativi di ripacchettamento.
        variant (bool): Se True, esegue la variante di Tabu Search.
    Returns:
        Tuple[Solution, bool, int]: La soluzione finale, un flag che indica se è completa e il numero
        di tentativi di ripacchettamento
    """

    # Funzione per resettare le strutture dati della tabu search
    def reset_tabu() -> None:
        nonlocal iteration, freq, tabu
        iteration = 0
        for i in range(len(best_sol.bins)):
            freq[i] = {}
            tabu[i] = {}

    # Funzione per verificare se una mossa è tabu
    def is_tabu(bin_idx: int, items_idxs: set[int]) -> bool:
        nonlocal iteration, tabu
        for item_idx in items_idxs:
            w = sol.weights[item_idx]
            # Se per il bin corrente esiste una tabu tenure per il peso w e questa è attiva
            if tabu[bin_idx].get(w, -1) >= iteration:
                return True
        return False

    # Funzione per aggiornare la tabu tenure
    def update_tabu(bin_idx: int, items_idxs: set[int]) -> None:
        nonlocal iteration, tabu, freq
        iteration += 1
        for item_idx in items_idxs:
            w = sol.weights[item_idx]
            # Incrementa la frequenza per il peso w
            freq_val = freq[bin_idx].get(w, 0) + 1
            freq[bin_idx][w] = freq_val
            # Imposta la tabu tenure in base alla frequenza 
            tabu[bin_idx][w] = iteration + math.ceil(freq_val / 2)

    
    if variant:
        # Variante di Tabu Search che limita le mosse che degradano il secondo obiettivo
        PQ = [(1,1), (2,1), (2,2), (3,1), (3,2)]
    else:
        PQ = [(0,1), (1,1), (2,1), (1,2), (2,2), (1,3), (3,1), (2,3), (3,2)]

    best_sol = init_sol.copy()
    sol = init_sol.copy()

    # Inizializza le strutture dati per la tabu search
    freq = [{} for _ in range(len(sol.bins))]
    tabu = [{} for _ in range(len(sol.bins))]

    # Inizializza le variabili per il ciclo di Tabu Search
    iteration = 0
    max_iterations = len(sol.bins) * sol.n
    complete_solution_found = False
    packing_attempts = attempts
    start_time = time()

    # Ciclo principale di Tabu Search
    while time() - start_time < time_limit and iteration < max_iterations:
        S_star = set()
        T_star = set()
        b_star = -1
        min_delta = sol.n * sum(sol.weights)

        # Per ogni bin, esamina tutte le mosse Swap(p, q) possibili
        for b_idx, bin_items in enumerate(sol.bins):
            for (p, q) in PQ:
                for S in combinations(bin_items, p):
                    if not is_tabu(b_idx, S):
                        for T in combinations(list(sol.trash), q):                            
                            current_bin_weight = sum(sol.weights[i] for i in bin_items)
                            new_bin_weight = current_bin_weight + sum(sol.weights[i] for i in T) - sum(sol.weights[i] for i in S) 
                            new_big_items_trash_count = sol.big_items_trash_count() + sum(1 for i in S if sol.weights[i] >= sol.capacity / 2) - sum(1 for i in T if sol.weights[i] >= sol.capacity / 2)
                            if new_bin_weight < sol.capacity and new_big_items_trash_count < 2:
                                # Calcola il delta dell'obiettivo
                                delta = sol.n * (sum(sol.weights[i] for i in S) - sum(sol.weights[i] for i in T)) - (len(S) - len(T))
                                if delta < min_delta:
                                    S_star = set(S)
                                    T_star = set(T)
                                    b_star = b_idx
                                    min_delta = delta
                                elif delta == min_delta:
                                    choice = random.randint(0,1)
                                    if choice == 0:
                                        S_star = set(S)
                                        T_star = set(T)
                                        b_star = b_idx
                                        min_delta = delta
        if len(T_star) == 0:
            break  # Termina quando non esiste una mossa consentita

        # Esegue la mossa swap
        swap(sol, b_star, S_star, T_star)
                
        # Se la soluzione è migliorata, aggiorna la migliore soluzione
        if sol.objective_value() <= best_sol.objective_value():
            best_sol = sol.copy()
            reset_tabu()

            # Se la soluzione è completa, esegui il ripacchettamento degli oggetti nel trash
            if best_sol.total_trash_weight() <= 2 * best_sol.capacity:
                packing_attempts += 1
                # Tentativo di ripacchettamento degli items nel trash
                trash_items = list(best_sol.trash)
                first_bin = pack_set(trash_items, best_sol.capacity, best_sol.weights)
                if sum(best_sol.weights[i] for i in first_bin) >= best_sol.total_trash_weight() - best_sol.capacity:
                    repacked = set(first_bin)
                    leftover = best_sol.trash - repacked
                    if len(leftover) > 0:
                        second_bin = list(leftover)
                        best_sol.bins.append(first_bin)
                        best_sol.bins.append(second_bin)
                    else: 
                        best_sol.bins.append(first_bin)
                    
                    best_sol.trash = set()
                    complete_solution_found = True
                    break
        else:
            update_tabu(b_star, T_star)

    return best_sol, complete_solution_found, packing_attempts

#@log_function_call
def CNS(sol: Solution, start_time: float, overall_time_limit: float) -> Tuple[Solution,bool]:
    """
    Algoritmo Consistent Neighborhood Search:
      - Esegue la procedura di Descent e Tabu Search in sequenza.
      - Termina quando non ci sono più miglioramenti o quando la soluzione è completa.
    Args:
        sol (Solution): Soluzione corrente.
        start_time (float): Tempo di inizio dell'esecuzione.
        overall_time_limit (float): Limite di tempo complessivo.
    Returns:
        Tuple[Solution, bool]: La soluzione finale e un flag che indica se è completa.
    """

    loops_count = 0
    packing_attempts = 0
    loops_without_solutions = 0
    complete_solution_found = False
    
    while time() - start_time < overall_time_limit and not complete_solution_found:
        if packing_attempts >= 100000:
            break
        if loops_count >= 10 and loops_without_solutions == loops_count:
            break
        
        loops_count += 1

        sol, complete_solution_found, packing_attempts = tabu_search(sol, TABU_TIME_LIMIT, packing_attempts, variant=False)
        
        if not complete_solution_found:
            sol, complete_solution_found, packing_attempts = tabu_search(sol, TABU_TIME_LIMIT, packing_attempts, variant=True)
        
        if not complete_solution_found:
            sol, complete_solution_found, packing_attempts = descent(sol, packing_attempts)

        if not complete_solution_found:
            loops_without_solutions +=1

    return sol, complete_solution_found


#############################################
# Algoritmo CNS_BP Complessivo
#############################################

def CNS_BP(n: int, c: int, weights: List[int], overall_time_limit: float = 60,seed: int = 1) -> Solution:
    random.seed(seed)
    """
    Algoritmo Consistent Neighborhood Search per il Bin Packing.
    Args:
        n (int): Numero totale di oggetti.
        c (int): Capacità del bin.
        weights (List[int]): Lista dei pesi degli oggetti.
        overall_time_limit (float): Limite di tempo complessivo.
        seed (int): Seme per la riproducibilità.
    Returns:
        Tuple[Solution, int]: La soluzione finale e il Lower Bound.
    """
    start_time = time()

    # Applica la riduzione preliminare
    remaining_items, fixed_bins = reduction(n, c, weights)

    # Costruisci la soluzione completa iniziale con First Fit (su ordine casuale)
    
    random.shuffle(remaining_items)
    bins = first_fit(remaining_items, weights, c)
    init_sol = Solution(bins, set(), c, weights)

    # Calcola Lower Bound (LB)
    LB = math.ceil(sum(weights[i] for i in remaining_items) / c)

    m = len(init_sol.bins)
    current_sol = init_sol.copy()
    
    # Ciclo principale per ridurre il numero di bin
    while m > LB and time() - start_time < overall_time_limit:
        m-=1
        # Rimuovi dalla soluzione corrente gli ultimi due bins e l'ultimo per il quale 
        # non si hanno più di 2 big items i (w(i) <= c) nel trash oppure il terzultimo
        remaining_bins = current_sol.bins[:-2]
        removed_bins = current_sol.bins[-2:]
        removed_items = set(i for b in removed_bins for i in b )
        
        for b in reversed(remaining_bins):
            candidate_trash = current_sol.trash.union(set(b)).union(removed_items)
            if sum(1 for i in candidate_trash if weights[i] >= c/2) <= 2:
                removed_bins.append(b)
                break
        else:
            candidate_trash = current_sol.trash.union(set(remaining_bins[-1])).union(removed_items)
            removed_bins.append(remaining_bins[-1])
        
        new_bins = [b for b in current_sol.bins if b not in removed_bins]
        new_trash = candidate_trash
        
        # Costruisci la soluzione parziale con m - 2 
        partial_sol = Solution(new_bins, new_trash, c, weights)

        if not partial_sol.check_trash_big_items() or not partial_sol.check_bins_capacity():
            break
        
        # Applica Local Search sulla soluzione parziale
        candidate_sol, complete_solution_found = CNS(partial_sol, start_time, overall_time_limit)
        
        # Se non è possibile trovare una soluzione completa con m - 2
        # restituisci esci dal ciclo
        if not complete_solution_found:
            break
        
        # Aggiorna la soluzione corrente
        current_sol = candidate_sol.copy()

    # Reinserisci i bins fissati dalla riduzione iniziale nell'ultima soluzione completa
    current_sol.bins.extend(fixed_bins)

    return current_sol

def run_instance(file_path: str, overall_time_limit: float, seed: int) -> Dict[str, Any]:
    """
    Esegue l'algoritmo CNS_BP su un'istanza specifica e restituisce i risultati.
    Args:
        file_path (str): Percorso del file di istanza.
        overall_time_limit (float): Limite di tempo complessivo.
        seed (int): Seme per la riproducibilità.
    Returns:
        Dict[str, Any]: Dizionario contenente i risultati dell'esperimento.
    """
    
    n, capacity, weights = read_instance(file_path)
    instance_name = os.path.basename(file_path)

    start_time = perf_counter()
    solution, LB = CNS_BP(n, capacity, weights, overall_time_limit, seed)
    exec_time = perf_counter() - start_time
    
    assert solution.validate_assignment(), f"Invalid weights assignment for instance {instance_name}!"

    sheet_name = os.path.dirname(file_path).split("/")[-1]
    df = pd.read_excel('./solutions/Solutions.xlsx', sheet_name=sheet_name)
    
    
    filter = df[df['Name'] == instance_name]['Best LB']

    # Verifica se il filtro ha restituito almeno una riga
    if not filter.empty:
        # Estrai il primo valore della Series
        opt_value = filter.iloc[0]
        # Confronta la lunghezza di solution.bins con il valore estratto
        if len(solution.bins) < opt_value:
            raise Exception("Soluzione sotto l'ottimo!")
        opt = len(solution.bins) == opt_value
    else:
        # Gestisci il caso in cui 'instance_name' non è presente nel DataFrame
        opt = False
        print(f"Attenzione: '{instance_name}' non trovato nel DataFrame.")

    return {
        "instance": instance_name,
        "n": n,
        "c": capacity,
        "num_bins": len(solution.bins),
        "execution_time": exec_time,
        "opt": opt,
        "opt_diff": len(solution.bins) - opt_value,
        "LB": LB,
        "opt_LB_diff": LB - opt_value
    }



def run_experiments(input_path: str, overall_time_limit: float, output_file: str, seed: int) -> None:
    """
    Esegue l'algoritmo CNS_BP su un insieme di istanze e salva i risultati in un file CSV.
    Args:
        input_path (str): Percorso della cartella contenente le istanze.
        overall_time_limit (float): Limite di tempo complessivo.
        output_file (str): Percorso del file CSV di output.
        seed (int): Seme per la riproducibilità.
    """
    
    # Verifica se il percorso di input è una cartella
    if os.path.isdir(input_path):

        # Se il file di output non esiste, crea un nuovo file e scrivi l'header
        if not os.path.isfile(output_file):
            with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
                fieldnames = ["instance", "n", "c", "num_bins", "execution_time","opt","opt_diff","LB","opt_LB_diff"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
        
        # Per ogni istanza, esegui l'algoritmo CNS_BP e scrivi i risultati nel file di output
        for filename in tqdm(sorted_dir_list(input_path), desc=f"Processing instances"):
            already_processed = False
            # Verifica se l'istanza è già stata processata
            if os.path.isfile(output_file):
                with open(output_file,"r",encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[0] == filename:
                            already_processed = True
                            print(f"Skipping instance {filename} (already processed)")
                            break
            if already_processed:
                continue
            
            # Esegui l'algoritmo CNS_BP sull'istanza corrente
            file_path = os.path.join(input_path, filename)
            print(f"Processing instance {filename}")
            if os.path.isfile(file_path):
                try:
                    result = run_instance(file_path, overall_time_limit, seed)
                except Exception as e:
                    print(f"Error processing instance {filename}: {e}")
                    print(f"Do you want to proceed to the next instance? (y/n)")
                    choice = input().lower()
                    if choice != 'y':
                        break
                    continue
                # Scrivi i risultati nel file di output
                with open(output_file, "a", newline='', encoding="utf-8") as csvfile:
                    print("Writing results to file...")
                    fieldnames = ["instance", "n", "c", "num_bins", "execution_time","opt","opt_diff","LB","opt_LB_diff"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(result)
    else:
        print(f"Path {input_path} is not a directory.")
        return
   

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNS_BP experimental framework for Bin Packing."
    )
    parser.add_argument("-i","--instance", type=str,
                        help="Path to an instance file.")
    parser.add_argument("-f","--folder", type=str,
                        help="Path to a folder containing instance files.")
    parser.add_argument("-t","--time_limit", type=float, default=60,
                        help="Overall time limit for each instance (seconds).")
    parser.add_argument("-o","--output", type=str, default="results.csv",
                        help="Output CSV file for experiment results.")
    parser.add_argument("-s", "--seed", type=int, default=1,
                        help="Seed for random number generator.")
    args = parser.parse_args()


    if args.folder:
        run_experiments(args.folder, args.time_limit, args.output, args.seed)
    elif args.instance:
        result = run_instance(args.instance, args.time_limit, args.seed)
        print(f"Instance: {args.instance}")
        print(f"Number of items (n): {result['n']}")
        print(f"Bin capacity (c): {result['c']}")
        print(f"Number of bins: {result['num_bins']}")
        print(f"Execution time (s): {result['execution_time']}")
        print(f"Optimal: {result['opt']}")
        print(f"Optimal difference: {result['opt_diff']}")
        print(f"Seed: {args.seed}")
    else:
        print("Please provide either an instance file (--instance) or a folder of instances (--folder).")

if __name__ == "__main__":
    main()   
