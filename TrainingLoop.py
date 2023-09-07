import Classes
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import gc
# import wandb

# Basic admin
np.random.seed(seed = 1234)

current_date = datetime.now()
formatted_date = current_date.strftime('%m-%d-%H')

# We define the parameters of the system
D_s = 200 # Dimension of the latent environmental state
D_o = 10 # Dimension of the observation space
D_a = 10 # Dimension of the action space
D_z = 40 # Dimension of the internal latent space
D_h = 80 # Dimension of the hidden layer in the network
dt = 0.1 # Time-step parameter
T_prob = 0.002 # Termination probability for each time-step
lr = 0.001 # Learning rate for the network
Total_steps = 150000 # The total number of training steps
Update_steps = 50 # Number of steps to perform between each update
Num_of_agents = 40 # Number of independent agent environment interactions

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# DEVICE = "cpu" 

print("Device:", DEVICE)

Reward_profile = torch.zeros(Total_steps) # We will use this to store the reward profile
Rewards_low_pass_filter = torch.zeros(Total_steps + 1) # We will use this to store the low pass filtered reward profile
Rewards_decay = 0.001 # We will use this to decay the low pass filtered reward profile
Rewards_decay_norm = 0 # We will use this to normalise the low pass filtered reward profile

# Create the environments and agents
print("Creating agents and environments.")
agent = Classes.Agent(D_a, D_o, D_z, D_h, dt, Num_of_agents, lr)
environment = Classes.Environment(D_s, D_o, D_a, dt, Num_of_agents)
actions = torch.zeros(Num_of_agents, D_a, device=DEVICE)
observations = torch.zeros(Num_of_agents, D_o, device=DEVICE)
rewards = torch.zeros(Num_of_agents, 1, device=DEVICE)

# Here we run the training loop for our agents

# Take the initial action

actions = agent.sample_action() # Sample action A_0
agent.compute_value()
    
# The training loop
print("Starting training loop.")
for step in range(1,Total_steps + 1):
    
    observations, rewards = environment.step(actions) # Get O_t, R_t given action A_{t-1}
    actions = agent.turn(observations, rewards, actions) # Given A_{t-1}, O_t, and R_t, compute A_t, g_{t-1}, and z_t

    # We randomly terminate episodes, and start the agent in a newly initialised environment
    if np.random.binomial(1, T_prob):
        environment.reset() # We reset the environment
        agent.reset() # We reset the agent's internal state
        gc.collect() # We perform garbage collection to free up memory
        if DEVICE == "cuda":   
            torch.cuda.empty_cache() # We empty the GPU cache
            # print(torch.cuda.memory_summary(device=None, abbreviated=True))
        actions = agent.sample_action() 
        agent.compute_value()
        print("Episode Terminated at time t =", step)

    Reward_profile[step - 1] = rewards.mean()
    Rewards_decay_norm = Rewards_decay_norm + Rewards_decay*(1 - Rewards_decay_norm)
    Rewards_low_pass_filter[step] = Rewards_low_pass_filter[step - 1] + (Rewards_decay/Rewards_decay_norm)*( rewards.mean() - Rewards_low_pass_filter[step - 1] ) 
        
    # Perform an update if we have reached the appropriate number of steps
    if step % Update_steps == 0:
        agent.update_weights(num_of_steps=Update_steps)
        agent.sever()
        agent.update_EntGrad()
        gc.collect() # We perform garbage collection to free up memory
        if DEVICE == "cuda":
            torch.cuda.empty_cache() # We empty the GPU cache
        print("Update performed at time t =", step)

# Create save_path which directs to the place we wish to save our results. 
current_directory = os.getcwd()
save_path = os.path.join(current_directory,'Results')
os.makedirs(save_path, exist_ok = True)

# We now save the network weights.
weights_file_name = f'Network_weights_{formatted_date}.pth'
weights_save_path = os.path.join(save_path, weights_file_name)
torch.save(agent.Layers.state_dict(), weights_save_path)

# We next save the environment parameters.
env_param_dict = {'P':environment.P, 'Q':environment.Q, 'K':environment.K, 'rho':environment.rho}
env_file_name = f'Environment_dict_{formatted_date}.pth'
env_save_path = os.path.join(save_path, env_file_name)
torch.save(env_param_dict, env_save_path)

# We next save the reward profile
reward_file_name = f'Rewards_{formatted_date}.pth'
reward_save_path = os.path.join(save_path, reward_file_name)
torch.save(Reward_profile, reward_save_path)

# Next, create a plot of the reward profile after filtering. 
# First we process the low-pass filtered reward profile by removing the first entry
Rewards_low_pass_filter = Rewards_low_pass_filter[1:]
fig, ax = plt.subplots(figsize = (8,5))
ax.plot(Rewards_low_pass_filter)
ax.tick_params(labelsize = 18)
ax.xlabel('Training step', fontsize = 20)
ax.ylabel('Reward running average', fontsize = 20)
ax.title('Reward profile', fontsize = 20)
fig.savefig(os.path.join(save_path, f'Reward_profile_{formatted_date}.pdf'), bbox_inches='tight')

