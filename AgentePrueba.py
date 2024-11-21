import numpy as np
import gym

class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min  # Asignar epsilon_min
        self.q_table = np.zeros(state_space + (action_space.n,))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action])
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def discretize_state(state):
    state = np.array(state) if not isinstance(state, np.ndarray) else state
    
    bins = [np.linspace(-2.4, 2.4, num=24),
            np.linspace(-2.0, 2.0, num=24),
            np.linspace(-0.5, 0.5, num=24),
            np.linspace(-0.5, 0.5, num=24)]
    
    state_indices = [np.digitize(state[i], bins[i]) - 1 for i in range(len(state))]
    state_indices = np.clip(state_indices, 0, np.array([24] * len(state)) - 1)
    
    return tuple(state_indices)

# Crear el entorno
env = gym.make('CartPole-v1', render_mode='human')

# Crear el agente
agent = QLearningAgent(state_space=(24, 24, 24, 24), action_space=env.action_space)

# Entrenamiento del agente
for episode in range(1000):
    observation, _ = env.reset()
    state = discretize_state(observation)
    total_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated  # Combina las condiciones de terminado
        next_state = discretize_state(next_observation)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    
    agent.decay_epsilon()
    print(f'Episode {episode} - Total Reward: {total_reward}')

env.close()