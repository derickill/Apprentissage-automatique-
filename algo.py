
import subprocess
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# --- CONFIGURATION GRAPHIQUE COMPACTE ET FLUIDE ---
plt.ion()
fig, ax = plt.subplots(figsize=(7, 4)) 
ax.set_xlim(155, 300)
ax.set_ylim(-110, 110) # Ajusté pour voir les punitions et récompenses
ax.set_title("Apprentissage par Renforcement (Q-Learning) - Dino", fontsize=10)
ax.set_xlabel("Distance de saut (d)")
ax.set_ylabel("Qualité de l'action (Q-Value)")

# On utilise des deques pour ne garder que les 100 derniers points (évite les lags)
MAX_POINTS = 100
x_data = deque(maxlen=MAX_POINTS)
y_data = deque(maxlen=MAX_POINTS)

# Objets graphiques légers
line, = ax.plot([], [], 'bo', markersize=4, alpha=0.5, label="Essais récents")
best_point, = ax.plot([], [], 'r*', markersize=12, label="Meilleure distance")
ax.legend(loc='upper right', fontsize='small')

# --- LOGIQUE MARKOVIENNE (Q-LEARNING) ---
ACTIONS = np.arange(160, 291, 1) # Espace d'états/actions discrétisé
q_table = {a: 0.0 for a in ACTIONS} # Table de connaissance de l'agent

# Paramètres de l'agent
rl_config = {
    'alpha': 0.3,      # Vitesse d'apprentissage
    'epsilon': 0.9,    # Taux d'exploration (commence haut)
    'best_d': 160,
    'best_q': -1000
}

# --- LANCEMENT DU PROCESSUS C ---
# Assurez-vous que le chemin est correct pour votre machine
process = subprocess.Popen(
    [r"C:\Users\attou\OneDrive\Desktop\welcome_C\projet_ml\x64\Debug\projet_ml.exe"],
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    text=True, 
    bufsize=1
)

def update_q_learning(d_brute, r_brute):
    """Calcule la vraie récompense et met à jour la Q-Table"""
    # 1. Correction de la logique de récompense (Reward Engineering)
    # Si r est élevé (ex: 50), c'est une collision -> Punition
    # Si r est faible (ex: 20), c'est un succès -> Récompense
    if r_brute >= 45:
        reward = -100 # Grosse punition pour l'échec
    else:
        reward = 100 - r_brute # Plus r est petit, plus le saut est propre

    # 2. Trouver l'action la plus proche dans notre table discrète
    action_idx = min(ACTIONS, key=lambda x: abs(x - d_brute))
    
    # 3. Équation de Bellman simplifiée (Mise à jour de la valeur d'état-action)
    # Q(s,a) = Q(s,a) + alpha * (R - Q(s,a))
    old_q = q_table[action_idx]
    q_table[action_idx] = old_q + rl_config['alpha'] * (reward - old_q)
    
    return action_idx, reward

try:
    while True:
        line_input = process.stdout.readline()
        if not line_input: break
        line_input = line_input.strip()

        # CAS 1 : Réception des données (Distance et Récompense brute)
        if " " in line_input:
            try:
                d_raw, r_raw = map(float, line_input.split())
                
                # Apprentissage
                idx, real_reward = update_q_learning(d_raw, r_raw)
                
                # Mise à jour des données pour le graph
                x_data.append(d_raw)
                y_data.append(q_table[idx]) # On affiche la valeur apprise (Q-Value)
                
                # Suivi de la meilleure performance
                if q_table[idx] > rl_config['best_q']:
                    rl_config['best_q'] = q_table[idx]
                    rl_config['best_d'] = d_raw
                
                # Rafraîchissement graphique fluide
                line.set_data(list(x_data), list(y_data))
                best_point.set_data([rl_config['best_d']], [rl_config['best_q']])
                
                plt.pause(0.0001)
                
            except ValueError:
                continue

        # CAS 2 : Le programme C demande une nouvelle action (Signal "1")
        elif line_input == "1":
            # Politique Epsilon-Greedy
            if rd.random() < rl_config['epsilon']:
                # Exploration : On teste une distance au hasard
                chosen_d = float(rd.choice(ACTIONS))
            else:
                # Exploitation : On prend la distance qui a la meilleure Q-Value
                chosen_d = float(max(q_table, key=q_table.get))
            
            # Envoi au programme C
            process.stdin.write(f"{chosen_d}\n")
            process.stdin.flush()
            
            # Réduction progressive de l'exploration (Epsilon Decay)
            rl_config['epsilon'] = max(0.05, rl_config['epsilon'] * 0.99)

except KeyboardInterrupt:
    print("\nEntraînement interrompu par l'utilisateur.")
finally:
    process.terminate()
    plt.close()