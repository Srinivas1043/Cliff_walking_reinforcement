import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def init_table(rows = 9, cols = 22):
    
    q_table = np.zeros((4, cols * rows))
    action_dict =  {"UP": q_table[0, :],"LEFT": q_table[1, :], "RIGHT": q_table[2, :], "DOWN": q_table[3, :]}
    
    return q_table


def epsilon_greedy_policy(state, q_table, epsilon = 0.1):
    decide_explore_exploit  = np.random.random()
    
    if(decide_explore_exploit < epsilon):
        action = np.random.choice(4) 
    else:
        action = np.argmax(q_table[:, state])
        
    return action
    


def move_agent(agent, action):

   
    (posX , posY) = agent
   
    if ((action == 0) and posX > 0):
        posX = posX - 1
    
    if((action == 1) and (posY > 0)):
        posY = posY - 1
    
    if((action == 2) and (posY < 21)):
        posY = posY + 1
   
    if((action) == 3 and (posX < 8)):
        posX = posX + 1
    agent = (posX, posY)
    
    return agent


def get_state(agent, q_table):
    
    (posX , posY) = agent
    
    state = 22 * posX + posY
    
    state_action = q_table[:, int(state)]
    maximum_state_value = np.amax(state_action) 
    return state, maximum_state_value

def get_reward(state):
   
    
    game_end = False
    reward = -1
    if(state == 197):
        game_end = True
        reward = 20
    if(state >= 187 and state <= 196):
        game_end = True
        reward = -100

    return reward, game_end

def update_qTable(q_table, state, action, reward, next_state_value, gamma_discount = 0.9, alpha = 0.5):

    update_q_value = q_table[action, state] + alpha * (reward + (gamma_discount * next_state_value) - q_table[action, state])
    q_table[action, state] = update_q_value

    return q_table    

def qlearning(num_episodes = 500, gamma_discount = 0.9, alpha = 0.5, epsilon = 0.1):

   
    reward_cache = list()
    step_cache = list()
    q_table = init_table()
    agent = (8, 0) 
    for episode in range(0, num_episodes):
        env = np.zeros((9, 22))
        env = visited_env(agent, env)
        agent = (8, 0) 
        game_end = False
        reward_cum = 0 
        step_cum = 0 
        while(game_end == False):
           
            state, _ = get_state(agent, q_table)
          
            action = epsilon_greedy_policy(state, q_table)
           
            agent = move_agent(agent, action)
            step_cum += 1
            env = visited_env(agent, env)
            next_state, max_next_state_value = get_state(agent, q_table)
           
            reward, game_end = get_reward(next_state)
            reward_cum += reward 
           
            q_table = update_qTable(q_table, state, action, reward, max_next_state_value, gamma_discount, alpha)
          
            state = next_state
        reward_cache.append(reward_cum)
        if(episode > 498):
            print("Agent trained with Q-learning after 500 iterations")
            print(env) 
        step_cache.append(step_cum)
    return q_table, reward_cache, step_cache

def sarsa(num_episodes = 500, gamma_discount = 0.9, alpha = 0.5, epsilon = 0.1):
    
   
    q_table = init_table()
    step_cache = list()
    reward_cache = list()
   
    for episode in range(0, num_episodes):
        agent = (8, 0)
        game_end = False
        reward_cum = 0
        step_cum = 0 
        env = np.zeros((9, 22))
        env = visited_env(agent, env)
        
        state, _ = get_state(agent, q_table)
        action = epsilon_greedy_policy(state, q_table)
        while(game_end == False):
           
            agent = move_agent(agent, action)
            env = visited_env(agent, env)
            step_cum += 1
            
            next_state, _ = get_state(agent, q_table)
           
            reward, game_end = get_reward(next_state)
            reward_cum += reward 
            
            next_action = epsilon_greedy_policy(next_state, q_table)
           
            next_state_value = q_table[next_action][next_state] 
            q_table = update_qTable(q_table, state, action, reward, next_state_value, gamma_discount, alpha)
            
            state = next_state
            action = next_action 
        reward_cache.append(reward_cum)
        step_cache.append(step_cum)
        if(episode > 498):
            print(" SARSA after 500 iterations")
            print(env) 
    return q_table, reward_cache, step_cache

def visited_env(agent, env):
  
    (posY, posX) = agent
    env[posY][posX] = 1
    return env
    
    
def retrieve_environment(q_table, action):
  
    env = q_table[action, :].reshape((4, 12))
    print(env) 
    
def plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA):
   
    cum_rewards_q = []
    rewards_mean = np.array(reward_cache_qlearning).mean()
    rewards_std = np.array(reward_cache_qlearning).std()
    count = 0 
    cur_reward = 0 
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if(count == 10):
           
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_q.append(normalized_reward)
            cur_reward = 0
            count = 0
            
    cum_rewards_SARSA = []
    rewards_mean = np.array(reward_cache_SARSA).mean()
    rewards_std = np.array(reward_cache_SARSA).std()
    count = 0
    cur_reward = 0 
    for cache in reward_cache_SARSA:
        count = count + 1
        cur_reward += cache
        if(count == 10):
            
            normalized_reward = (cur_reward - rewards_mean)/rewards_std
            cum_rewards_SARSA.append(normalized_reward)
            cur_reward = 0
            count = 0      
    
    plt.plot(cum_rewards_q, label = "q_learning")
    plt.plot(cum_rewards_SARSA, label = "SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Convergence of Cumulative Reward")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    
def plot_number_steps(step_cache_qlearning, step_cache_SARSA):
    
    cum_step_q = []
    steps_mean = np.array(step_cache_qlearning).mean()
    steps_std = np.array(step_cache_qlearning).std()
    count = 0 
    cur_step = 0 
    for cache in step_cache_qlearning:
        count = count + 1
        cur_step += cache
        if(count == 10):
            
            normalized_step = (cur_step - steps_mean)/steps_std
            cum_step_q.append(normalized_step)
            cur_step = 0
            count = 0
            
    cum_step_SARSA = []
    steps_mean = np.array(step_cache_SARSA).mean()
    steps_std = np.array(step_cache_SARSA).std()
    count = 0 
    cur_step = 0 
    for cache in step_cache_SARSA:
        count = count + 1
        cur_step += cache
        if(count == 10):
            
            normalized_step = (cur_step - steps_mean)/steps_std
            cum_step_SARSA.append(normalized_step)
            cur_step = 0
            count = 0      
    
    plt.plot(cum_step_q, label = "q_learning")
    plt.plot(cum_step_SARSA, label = "SARSA")
    plt.ylabel('Number of iterations')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Iteration number untill game ends")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    

    
def plot_qlearning_smooth(reward_cache):
   
    mean_rev = (np.array(reward_cache[0:21]).sum())/10
   
    cum_rewards = [mean_rev] * 10
    idx = 0
    for cache in reward_cache:
        cum_rewards[idx] = cache
        idx += 1
        smooth_reward = (np.array(cum_rewards).mean())
        cum_rewards.append(smooth_reward)
        if(idx == 10):
            idx = 0
        
    plt.plot(cum_rewards)
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning  Convergence of Cumulative Reward")
    plt.legend(loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()

def generate_heatmap(q_table):
   
    import seaborn as sns; sns.set()
    
    data = np.mean(q_table, axis = 0)
    print(data)
    data = data.reshape((9, 22))
    ax = sns.heatmap(np.array(data))
    return ax
    
def main():
    #SARSA
    q_table_SARSA, reward_cache_SARSA, step_cache_SARSA = sarsa()
    # QLEARNING
    q_table_qlearning, reward_cache_qlearning, step_cache_qlearning = qlearning()
    plot_number_steps(step_cache_qlearning, step_cache_SARSA)
    # Visualize the result
    plot_cumreward_normalized(reward_cache_qlearning,reward_cache_SARSA)
    
    # generate heatmap
    print("Visualize environment Q-learning")
    ax_q = generate_heatmap(q_table_qlearning)
    print(ax_q)
    
    print("Visualize SARSA")
    ax_SARSA = generate_heatmap(q_table_SARSA)
    print(ax_SARSA)
    
    # Debug method giving information about what are some states for environment
    want_to_see_env = False
    if(want_to_see_env):
        print("UP")
        retrieve_environment(q_table_qlearning, 0)
        print("LEFT")
        retrieve_environment(q_table_qlearning, 1)
        print("RIGHT")
        retrieve_environment(q_table_qlearning, 2)
        print("DOWN")
        retrieve_environment(q_table_qlearning, 3)
    want_to_see_env = False
    if(want_to_see_env):
        print("UP")
        retrieve_environment(q_table_SARSA, 0)
        print("LEFT")
        retrieve_environment(q_table_SARSA, 1)
        print("RIGHT")
        retrieve_environment(q_table_SARSA, 2)
        print("DOWN")
        retrieve_environment(q_table_SARSA, 3)
    
if __name__ == "__main__":
   
    main()