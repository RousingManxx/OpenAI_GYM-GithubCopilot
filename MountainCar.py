import gym
import numpy as np
import matplotlib.pyplot as plt

# Crear el entorno
env = gym.make('MountainCar-v0')

# Parámetros de aprendizaje
alpha = 0.1  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Tasa de exploración inicial
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 100  # Número de episodios

# Inicializar la tabla Q
state_space = (env.observation_space.high - env.observation_space.low) * np.array([10, 100])
state_space = np.round(state_space, 0).astype(int) + 1
Q = np.random.uniform(low=-1, high=1, size=(state_space[0], state_space[1], env.action_space.n))

def discretize_state(state):
    state_adj = (state - env.observation_space.low) * np.array([10, 100])
    return np.round(state_adj, 0).astype(int)

# Listas para monitoreo
episode_rewards = []

# Entrenamiento del agente
for episode in range(num_episodes):
    state = discretize_state(env.reset()[0])  # Extraer la observación del diccionario
    total_reward = 0
    done = False
    
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state[0], state[1]])

        result = env.step(action)
        next_state = result[0]
        reward = result[1]
        done = result[2]

        next_state = discretize_state(next_state)

        if done and next_state[0] >= env.goal_position:
            Q[state[0], state[1], action] = reward
        else:
            Q[state[0], state[1], action] += alpha * (reward + gamma * np.max(Q[next_state[0], next_state[1]]) - Q[state[0], state[1], action])

        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Reducir epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Guardar las recompensas
    episode_rewards.append(total_reward)
    
    if episode % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episodio: {episode}, Recompensa Total: {total_reward}, Recompensa Promedio: {avg_reward}")

# Evaluación del agente
env = gym.make('MountainCar-v0', render_mode='human')  # Activar la visualización
state = discretize_state(env.reset()[0])  # Extraer la observación del diccionario
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state[0], state[1]])
    result = env.step(action)
    next_state = result[0]
    reward = result[1]
    done = result[2]
    next_state = discretize_state(next_state)
    
    state = next_state
    total_reward += reward
    
    if done:
        break

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