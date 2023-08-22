import Classes
import torch
import numpy as np
import os
from datetime import datetime

current_date = datetime.now()
formatted_date = current_date.strftime('%m-%d-%H')

# We define the parameters of the system
D_s = 250 # Dimension of the latent environmental state
D_o = 10 # Dimension of the observation space
D_a = 10 # Dimension of the action space
D_z = 40 # Dimension of the internal latent space
D_h = 100 # Dimesnion of the hidden layer in the network
dt = 0.1 # Time-step parameter
T_prob = 0.002 # Termination probability for each time-step
lr = 0.001 # Learning rate for the network
Total_steps = 100000 # The total number of training steps
Update_steps = 100 # Number of steps to perform between each update
Num_of_agents = 64 # Number of independent agent environment interactions

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

#DEVICE = "cpu"

print("Device:", DEVICE)

Reward_profile = torch.zeros(Total_steps)

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
        environment.reset()
        agent.reset()
        actions = agent.sample_action()
        agent.compute_value()
        print("Episode Terminated at time t =", step)

    Reward_profile[step - 1] = rewards.mean()
        
    # Perform an update if we have reached the appropriate number of steps
    if step % Update_steps == 0:
        agent.update_weights(num_of_steps=Update_steps)
        agent.sever()
        agent.update_EntGrad()
        print("Update performed at time t =", step)

# We now save the network weights
current_directory = os.getcwd()
save_path = os.path.join(current_directory,'SavedNetworks')
os.makedirs(save_path, exist_ok = True)
file_name = f'Network_weights_{formatted_date}.pth'
save_path = os.path.join(save_path, file_name)
torch.save(agent.Layers.state_dict(), save_path)
