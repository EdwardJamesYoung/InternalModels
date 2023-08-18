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
Total_steps = 250 # The total number of training steps
Update_steps = 50 # Number of steps to perform between each update
Num_of_agents = 5 # Number of independent agent environment interactions
Terminate = [False] # Termination variables for each agent-environment pair

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

#DEVICE = "cpu"

print("Device:", DEVICE)

Reward_profile = torch.zeros(Total_steps)

# Create the environments and agents
print("Creating agents and environments.")
agents = [Classes.Agent(D_a, D_o, D_z, D_h, dt)]
environments = [Classes.Environment(D_s, D_o, D_a, dt)]
actions = [torch.zeros(D_a, device = DEVICE)]
observations = [torch.zeros(D_o, device = DEVICE)]
rewards = [0]
for kk in range(1, Num_of_agents):
    agents.append(agents[0].clone_agent())
    environments.append(environments[0].clone_environment())
    actions.append(torch.zeros(D_a, device = DEVICE))
    observations.append(torch.zeros(D_o, device = DEVICE))
    rewards.append(0)
    Terminate.append(False)

# Create the optimiser
print("Creating optimiser.")
optm = Classes.Network_optimiser(agents[0].Layers, 0.001)


# Here we run the training loop for our agents

# Take the initial action
for kk in range(Num_of_agents):
    actions[kk] = agents[kk].sample_action() # Sample action A_0
    agents[kk].compute_value()
    
# The training loop
print("Starting training loop.")
for step in range(1,Total_steps + 1):
    for kk in range(Num_of_agents):
        
        observations[kk], rewards[kk] = environments[kk].step(actions[kk]) # Get O_t, R_t given action A_{t-1}
        actions[kk] = agents[kk].turn(observations[kk], rewards[kk], actions[kk]) # Given A_{t-1}, O_t, and R_t, compute A_t, g_{t-1}, and z_t

        # We randomly terminate episodes, and start the agent in a newly initialised environment
        if np.random.binomial(1, T_prob):
            environments[kk].reset()
            agents[kk].reset()
            actions[kk] = agents[kk].sample_action()
            agents[kk].compute_value()
            print("Episode Terminated for agent",kk,"at time t =", step)

    Reward_profile[step - 1] = sum(rewards)/Num_of_agents
            
    # Perform an update if we have reached the appropriate number of steps
    if step % Update_steps == 0:
        grads = []
        for kk in range(Num_of_agents):
            grads.append(agents[kk].Total_gradients)
            
        optm.average_gradients(grads, Update_steps)
        optm.update()
        for kk in range(Num_of_agents):
            optm.copy_weights_to_agent(agents[kk])
            agents[kk].sever()
            agents[kk].update_EntGrad()
        print("Update performed at time t =", step)

print(Reward_profile)