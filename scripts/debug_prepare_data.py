import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Definizione connessioni COCO (per verificare se i dati sono COCO)
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms
    (11, 12), (5, 11), (6, 12),          # Torso
    (11, 13), (13, 15), (12, 14), (14, 16) # Legs
]

# Definizione connessioni H36M (per confronto)
H36M_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), # Legs
    (0, 7), (7, 8), (8, 9), (9, 10), # Spine/Head
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16) # Arms
]

def debug_data(data_path='data/processed/X.npy'):
    if not os.path.exists(data_path):
        print(f"File non trovato: {data_path}")
        # Prova percorso locale se esiste
        local_path = os.path.join(os.getcwd(), 'data', 'processed', 'X.npy')
        if os.path.exists(local_path):
            data_path = local_path
            print(f"Trovato in locale: {data_path}")
        else:
            return

    print("Caricamento dati...")
    X = np.load(data_path)
    print(f"Shape dati: {X.shape}") 
    # Atteso: (N, C, T, V) oppure (N, T, V, C) o simile.
    # Dataset.py si aspetta (N, C, T, V) di solito.
    
    sample = X[0] # Prendi il primo video
    
    # Adatta shape: vogliamo (V, 2) o (V, 3) per il plot
    # Se shape è (C, T, V) -> (3, 40, 17)
    if sample.shape[0] in [2, 3, 4]: 
        # (C, T, V)
        frame = sample[:, 0, :] # Primo frame: (C, V)
        x = frame[0, :]
        y = frame[1, :]
    elif sample.shape[-1] in [2, 3, 4]:
        # (T, V, C)
        frame = sample[0, :, :] # Primo frame
        x = frame[:, 0]
        y = frame[:, 1]
    else:
        print(f"Formato dati non riconosciuto: {sample.shape}")
        return

    # Visualizzazione
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Interpretazione COCO
    ax1.set_title("Se i dati sono COCO (Atteso)")
    ax1.scatter(x, -y) # -y per invertire asse immagine
    for i, j in COCO_CONNECTIONS:
        if i < len(x) and j < len(x):
            ax1.plot([x[i], x[j]], [-y[i], -y[j]], 'b-')
    ax1.set_aspect('equal')
    
    # Plot 2: Interpretazione H36M
    ax2.set_title("Se i dati sono H36M (Mismatch)")
    ax2.scatter(x, -y)
    for i, j in H36M_CONNECTIONS:
        if i < len(x) and j < len(x):
            ax2.plot([x[i], x[j]], [-y[i], -y[j]], 'r-')
    ax2.set_aspect('equal')
    
    print("\n--- ANALISI ---")
    print("Guarda i grafici:")
    print("1. Se il grafico BLU (Sinistra) sembra un umano -> Tutto OK (Dati sono COCO)")
    print("2. Se il grafico ROSSO (Destra) sembra un umano -> Mismatch (Dati sono H36M ma usati come COCO)")
    print("3. Se entrambi sono mostri -> Dati corrotti o altro formato")
    
    output_filename = 'debug_skeleton_viz.png'
    plt.savefig(output_filename)
    print(f"\n[INFO] Grafico salvato come '{output_filename}' nella cartella corrente.")
    # plt.show()

if __name__ == "__main__":
    # Cerca di capire dove sta X.npy
    # Di default cerca in data/processed relativo alla root
    debug_data()
