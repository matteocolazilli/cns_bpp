#!/usr/bin/env python3

from time import perf_counter
import os
import csv
import argparse
from typing import Dict, Any
from algos.auxiliary import read_instance, sorted_dir_list
from tqdm import tqdm
from algos.cns import CNS_BP
from algos.best_fit import best_fit
import pandas as pd

CNS = "cns"
BF = "bf"

def run_instance(algo: str, file_path: str, overall_time_limit: float, seed: int) -> Dict[str, Any]:
    """
    Esegue l'algoritmo scelto su un'istanza specifica e restituisce i risultati.
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
    if algo == CNS:
        solution = CNS_BP(n, capacity, weights, overall_time_limit, seed)
    elif algo == BF:
        solution = best_fit(n, capacity, weights)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    exec_time = perf_counter() - start_time
    
    assert solution.validate_assignment(), f"Invalid weights assignment for instance {instance_name}!"

    sheet_name = os.path.dirname(file_path).split("/")[1]
    df = pd.read_excel('./solutions/Solutions.xlsx', sheet_name=sheet_name)
    
    
    filter = df[df['Name'] == instance_name]['Best LB']

    # Verifica se il filtro ha restituito almeno una riga
    if not filter.empty:
        # Estrai il primo valore della Series
        opt_value = filter.iloc[0]
        # Confronta la lunghezza di solution.bins con il valore estratto
        if len(solution.bins) < opt_value:
            raise Exception("Soluzione sotto l'ottimo!")
    else:
        # Gestisci il caso in cui 'instance_name' non è presente nel DataFrame
        print(f"Attenzione: '{instance_name}' non trovato nel DataFrame.")

    return {
        "instance": instance_name,
        "n": n,
        "c": capacity,
        "num_bins": len(solution.bins),
        "execution_time": exec_time,
        "opt_diff": len(solution.bins) - opt_value,
    }



def run_experiments(algo: str, input_path: str, overall_time_limit: float, output_file: str, seed: int) -> None:
    """
    Esegue l'algoritmo scelto su un insieme di istanze e salva i risultati in un file CSV.
    Args:
        algo (str): Algoritmo da eseguire (CNS_BP o BestFit).
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
                fieldnames = ["instance", "n", "c", "num_bins", "execution_time", "opt_diff"]
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
            
            # Esegui l'algoritmo scelto sull'istanza corrente
            file_path = os.path.join(input_path, filename)
            print(f"Processing instance {filename}")
            if os.path.isfile(file_path):
                try:
                    result = run_instance(algo, file_path, overall_time_limit, seed)
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
                    fieldnames = ["instance", "n", "c", "num_bins", "execution_time", "opt_diff"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(result)
    else:
        print(f"Path {input_path} is not a directory.")
        return
   

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental framework for Bin Packing."
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
    parser.add_argument("-a", "--algo", type=str, default="CNS_BP",
                        help="Algorithm to run (cns or bf (BestFit)).",required=True,choices=[CNS,BF])
    args = parser.parse_args()


    if args.folder:
        run_experiments(args.algo, args.folder, args.time_limit, args.output, args.seed)
    elif args.instance:
        result = run_instance(args.algo,args.instance, args.time_limit, args.seed)
        print(f"Instance: {args.instance}")
        print(f"Number of items (n): {result['n']}")
        print(f"Bin capacity (c): {result['c']}")
        print(f"Number of bins: {result['num_bins']}")
        print(f"Execution time (s): {result['execution_time']}")
        print(f"Seed: {args.seed}")
    else:
        print("Please provide either an instance file (--instance) or a folder of instances (--folder).")

if __name__ == "__main__":
    main()   
