from typing import List, Set


class Solution:
    """
    Rappresenta una soluzione (completa o parziale) per il Bin Packing.
    
    Attributes:
        bins (List[List[int]]): Lista di bin, ognuno con la lista di indici degli oggetti assegnati.
        trash (Set[int]): Insieme degli oggetti non assegnati ("trash can").
        capacity (int): Capacità massima di ciascun bin.
        weights (List[int]): Lista dei pesi degli oggetti.
        n (int): Numero totale degli oggetti.
    """
    def __init__(self, bins: List[List[int]], trash: Set[int], capacity: int, weights: List[int]) -> None:
        self.bins = bins
        self.trash = set(trash)
        self.capacity = capacity
        self.weights = weights
        self.n = len(weights)
    
    def copy(self) -> "Solution":
        new_bins = [b[:] for b in self.bins]
        new_trash = set(self.trash)
        new_sol = Solution(new_bins, new_trash, self.capacity, self.weights)
        return new_sol
    
    def total_trash_weight(self) -> int:
        return sum(self.weights[i] for i in self.trash)
    
    def trash_count(self) -> int:
        return len(self.trash)
    
    def objective_value(self) -> int:
        """
        Funzione obiettivo da minimizzare:
          n * (peso totale nel trash) - (numero di oggetti nel trash)
        """
        return self.n * self.total_trash_weight() - self.trash_count()
    
    def check_bins_capacity(self) -> bool:
        """
        Verifica che in ogni bin la somma dei pesi non superi la capacità c.
        """
        for b in self.bins:
            if sum(self.weights[i] for i in b) > self.capacity:
                return False
        return True

    def big_items_trash_count(self):
        """
        Restituisce il numero di "big items" (oggetti con peso >= c/2) presenti nel trash.
        """

        return sum(1 for i in self.trash if self.weights[i] >= self.capacity / 2)

    def check_trash_big_items(self) -> bool:
        """
        Verifica che nel trash siano presenti al massimo 2 "big items"
        (oggetti con peso >= c/2).
        """
        count_big = sum(1 for i in self.trash if self.weights[i] >= self.capacity / 2)
        return count_big <= 2
    
    def __eq__(self, other: "Solution") -> bool:
        if not isinstance(other, Solution):
            return NotImplemented

        # Verifica capacità, pesi e trash
        if self.capacity != other.capacity or self.weights != other.weights or self.trash != other.trash:
            return False

        # Normalizza i bins: per ogni bin, ordina gli elementi e poi ordina la lista dei bins
        self_normalized = sorted(tuple(sorted(bin)) for bin in self.bins)
        other_normalized = sorted(tuple(sorted(bin)) for bin in other.bins)

        return self_normalized == other_normalized

    def validate_assignment(self) -> bool:
        """
        Verifica la validità dell'assegnazione degli oggetti ai contenitori (bins).
        Controlla che:
        - Tutti gli oggetti originali siano stati assegnati e che non ci siano oggetti extra.
        - Nessun contenitore superi la capacità massima.
        - Il contenitore di scarto (trash) sia vuoto.
        Returns:
            bool: True se l'assegnazione è valida, False altrimenti.        
        """
        assigned_items = [i for b in self.bins for i in b]
        original_items = set(range(self.n))
        
        if len(assigned_items) > len(original_items):
            extra_items = [i for i in assigned_items if i not in original_items]
            print(f"Errore: ci sono oggetti assegnati non validi: {extra_items}")
            return False
        
        elif len(assigned_items) < len(original_items):
            missing_items = [i for i in original_items if i not in assigned_items]
            print(f"Errore: ci sono oggetti non assegnati: {missing_items}")
            return False

        for b in self.bins:
            total_weight = sum(self.weights[i] for i in b)
            if total_weight > self.capacity:
                print(f"Errore: il bin {b} supera la capacità!")
                return False
            
        if len(self.trash) > 0:
            print(f"Errore: il trash can non è vuoto!")
            return False
        
        return True