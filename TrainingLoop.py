import Classes
import torch
import numpy as np


# We define the parameters of the system
D_s = 20 # Dimension of the latent environmental state
D_o = 5 # Dimension of the observation space
D_a = 4 # Dimension of the action space
D_z = 10 # Dimension of the internal latent space
D_h = 20 # Dimesnion of the hidden layer in the network
dt = 0.1 # Time-step parameter
T_prob = 0.01 # Termination probability for each time-step
lr = 0.001
Total_steps = 250 # The total number of training steps
Update_steps = 50 # Number of steps to perform between each update
Num_of_agents = 32 # Number of independent agent environment interactions
Terminate = False # Termination variables for each agent-environment pair

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

print(Reward_profile)