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
    parser.add_argument("-n", type=int, help="Numero di items per ogni istanza",required=True)
    parser.add_argument("-c", type=int, help="Capacità del bin",required=True)
    args = parser.parse_args()

    for i in range(1, args.m + 1):
        weights = generate_instance(args.l, args.u, args.n, args.c)
        weights.sort(reverse=True)
        filename = f"instance_{args.n}_{args.c}_{args.l}_{args.u}_{i}.txt"
        with open(filename, "w") as f:
            f.write(f"{args.n}\n")
            f.write(f"{args.c}\n")
            for w in weights:
                f.write(f"{w}\n")

    print(f"Generazione completata: {args.m} istanze create.")
    
if __name__ == "__main__":
    main()
