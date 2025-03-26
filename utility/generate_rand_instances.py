import os
import random
import argparse

def generate_instance(l, u, n, c):
    """
    Genera una lista di n pesi casuali compresi tra l*c e u*c.
    In questo esempio generiamo pesi interi, ma se serve puoi modificare
    il comportamento (ad esempio usando random.uniform per numeri float).
    """
    lower = int(l * c)
    upper = int(u * c)
    weights = [random.randint(lower, upper) for _ in range(n)]
    return weights

def main():
    parser = argparse.ArgumentParser(
        description="Genera m istanze per il problema del bin packing."
    )
    parser.add_argument("-m", type=int, help="Numero di istanze da generare", default=1)
    parser.add_argument("-l", type=float, help="Fattore inferiore per il peso (minore)",required=True,choices=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])   
    parser.add_argument("-u", type=float, help="Fattore superiore per il peso (maggiore)",required=True)
    parser.add_argument("-n", type=str, help="Intervallo per il numero di items per ogni istanza (es. 10-20)", required=True)
    parser.add_argument("-c", type=int, help="Capacità del bin",required=True)
    args = parser.parse_args()

    if args.u < args.l:
        parser.error("Il fattore superiore deve essere maggiore o uguale al fattore inferiore")


    output_dir = "instances/My_Randomly_Generated/"
    os.makedirs(output_dir, exist_ok=True)

    n_range = list(map(int, args.n.split('-')))
    for i in range(1, args.m + 1):
        n = random.randint(n_range[0], n_range[1])
        weights = generate_instance(args.l, args.u, n, args.c)
        weights.sort(reverse=True)
        filename = os.path.join(output_dir, f"instance_{n}_{args.c}_{args.l}_{args.u}_{i}.txt")
        with open(filename, "w") as f:
            f.write(f"{n}\n")
            f.write(f"{args.c}\n")
            for w in weights:
                f.write(f"{w}\n")

    print(f"Generazione completata: {args.m} istanze create in {output_dir}")
    
if __name__ == "__main__":
    main()
