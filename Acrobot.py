import gym
import numpy as np
import matplotlib.pyplot as plt

# Crear el entorno
env = gym.make('Acrobot-v1')
state_space = (20, 20, 20, 20, 20, 20)  # Ajustar el espacio de estados discretizado
action_space = env.action_space.n

# Parámetros de Q-learning
learning_rate = 0.1  # Ajustar la tasa de aprendizaje
discount_factor = 0.99  # Ajustar el factor de descuento
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Inicialización de la tabla Q
q_table = np.zeros(state_space + (action_space,))

# Función para discretizar el espacio continuo de estados
def discretize_state(state):
    if isinstance(state, dict):
        state = state['observation']
    state = np.array(state, dtype=np.float32)  # Convertir el estado a un array de NumPy
    bins = [np.linspace(-1, 1, num=state_space[i]) for i in range(len(state))]
    state_indices = [np.digitize(state[i], bins[i]) - 1 for i in range(len(state))]
    state_indices = np.clip(state_indices, 0, np.array(state_space) - 1)
    return tuple(state_indices)

# Listas para monitoreo
episode_rewards = []

# Entrenamiento del agente
for episode in range(10000):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state  # Extraer el estado si es un tuple
    state = state['observation'] if isinstance(state, dict) else state
    state = discretize_state(state)
    total_reward = 0
    done = False
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = next_state[0] if isinstance(next_state, tuple) else next_state  # Extraer el estado si es un tuple
        next_state = next_state['observation'] if isinstance(next_state, dict) else next_state
        next_state = discretize_state(next_state)
        
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += learning_rate * (reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action])
        
        state = next_state
        total_reward += reward
        
        if terminated or truncated:
            done = True
    
    # Reducir epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Guardar las recompensas
    episode_rewards.append(total_reward)
    
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episodio: {episode}, Recompensa Total: {total_reward}, Recompensa Promedio: {avg_reward}")

# Evaluación del agente
env = gym.make('Acrobot-v1', render_mode='human')  # Activar la visualización
state = env.reset()
state = state[0] if isinstance(state, tuple) else state  # Extraer el estado si es un tuple
state = state['observation'] if isinstance(state, dict) else state
state = discretize_state(state)
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[state])
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = next_state[0] if isinstance(next_state, tuple) else next_state  # Extraer el estado si es un tuple
    next_state = next_state['observation'] if isinstance(next_state, dict) else next_state
    next_state = discretize_state(next_state)
    
    state = next_state
    total_reward += reward
    
    if terminated or truncated:
        done = True

print(f"Recompensa Total en Evaluación: {total_reward}")

# Cerrar el entorno
env.close()

# Monitoreo y visualización
# Calcular recompensas promedio cada 100 episodios
episodes = list(range(0, 10000, 100))
average_rewards = [np.mean(episode_rewards[i:i + 100]) for i in range(0, len(episode_rewards), 100)]

# Graficar el progreso
plt.plot(episodes, average_rewards)
plt.xlabel('Episodios')
plt.ylabel('Recompensa Promedio')
plt.title('Progreso del Agente')
plt.grid(True)
plt.savefig('agent_progress.png')
plt.show()

# Guardar la tabla Q
np.save('q_table.npy', q_table)