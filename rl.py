import numpy as np 
import matplotlib.pyplot as plt


grid_size = 4
num_states = grid_size*grid_size
num_actions = 4
def get_new_state(state,action):
    row,col=divmod(state,grid_size) 
    if action == 0 and row > 0:
        row-=1
    elif action == 1 and row < grid_size-1:
        col+=1
    elif action == 2 and col >0:
        col-=1
    elif action == 3 and col < grid_size-1:
        col+=1
    return row*grid_size+col
def get_reward(state):
    return 10 if state == num_states-1 else-1
 
q_table = np.zeros(num_states,num_actions)
learning_rate= 0.1
discount_factor=0.6
epsilon=0.1 
# train the agent
num_episodes = 1000
rewards = []
for episode in range(num_episodes):
    state = 0  
    total_reward = 0
    while state != num_states - 1:  
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)  
        else:
            action = np.argmax(q_table[state])

        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state
        total_reward += reward
    rewards.append(total_reward)
plt.plot(rewards)
plt.title('Rewards Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
state = 0
path = [state]
while state != num_states - 1:
    action = np.argmax(q_table[state])
    state = get_next_state(state, action)
    path.append(state)
print("Optimal Path:", path)