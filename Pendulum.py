import gym
import numpy as np
import matplotlib.pyplot as plt

# Crear el entorno
env = gym.make('Pendulum-v1')

# Parámetros de aprendizaje
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Tasa de exploración inicial
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 10000  # Número de episodios

# Discretización del espacio de estados
state_bins = [
    np.linspace(-1, 1, 20),  # theta
    np.linspace(-1, 1, 20),  # theta_dot
    np.linspace(-8, 8, 20)   # torque
]

def discretize_state(state):
    state_indices = []
    for i in range(len(state)):
        state_indices.append(np.digitize(state[i], state_bins[i]) - 1)
    return tuple(state_indices)

# Inicializar la tabla Q
state_space_size = tuple(len(bins) for bins in state_bins)
action_space_size = 3  # Discretizamos el espacio de acción en 3 acciones: -2, 0, 2
Q = np.random.uniform(low=-1, high=1, size=state_space_size + (action_space_size,))

# Listas para monitoreo
episode_rewards = []

# Entrenamiento del agente
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0])  # Extraer la observación de la tupla
    total_reward = 0
    done = False
    
    while not done:
        if np.random.rand() < epsilon:
            action_index = np.random.choice(action_space_size)
        else:
            action_index = np.argmax(Q[state])

        action = np.array([action_index * 2 - 2])  # Convertir índice a acción continua (-2, 0, 2)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize_state(next_state)

        Q[state][action_index] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action_index])

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
env = gym.make('Pendulum-v1', render_mode='human')  # Activar la visualización
state = discretize_state(env.reset()[0])  # Extraer la observación de la tupla
done = False
total_reward = 0

while not done:
    action_index = np.argmax(Q[state])
    action = np.array([action_index * 2 - 2])  # Convertir índice a acción continua (-2, 0, 2)
    next_state, reward, terminated, truncated, _ = env.step(action)
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
episodes = list(range(0, num_episodes, 100))
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
np.save('q_table.npy', Q)