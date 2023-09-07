import numpy as np
from torch import nn
import torch.nn.functional as F
import copy
import torch
import os
torch.autograd.set_detect_anomaly(True)

# Basic admin
np.random.seed(seed = 1234)

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# DEVICE = "cpu"

# Build the environment class
class Environment(object):
    def __init__(self, D_s, D_o, D_a, dt, num_of_agents):
        self.D_s = D_s # Dimension of the latent environmental state
        self.D_o = D_o # Dimension of the observation space
        self.D_a = D_a # Dimension of the action space
        self.dt = dt # Time-step parameter
        self.N = num_of_agents # Number of agents in the environment
        
        # We randomly generate the dynamics of the environment 
        self.Q = dt*np.sqrt(2/(D_s + D_a))*torch.randn(D_a, D_s, device = DEVICE)
        self.K = np.sqrt(2/(D_s + D_o))*torch.randn(D_s, D_o, device = DEVICE)
        self.rho = (1/np.sqrt(D_s))*torch.randn(D_s, device = DEVICE)
        
        # Sample P such that the leading eigenvalue has absolute value below 1
        eigV = 1.1*torch.ones(D_s)
        while not (eigV <= 1).all():
            self.P = (1 - dt)*torch.eye(D_s, device = DEVICE) + (dt/np.sqrt(D_s))*torch.randn(D_s, D_s, device = DEVICE)
            eigV, _ = torch.linalg.eig(self.P)
            eigV = abs(eigV)
        
        # Define the state vector. The first index is the agent number, the second is the state dimension.
        self.s = (0.4/np.sqrt(self.D_s))*torch.randn(self.N, self.D_s, device = DEVICE)
    
    def step(self, a):
        # Given an action, update the state vector.
        self.s = (self.s@self.P + a@self.Q).detach()
        
        # Return the observation of the next state, along with the reward associated with that state. 
        o = (self.s@self.K).detach()
        r = (self.s@self.rho).detach()
        return o, r
    
    def reset(self):
        self.s = (0.4/np.sqrt(self.D_s))*torch.randn(self.N, self.D_s, device = DEVICE)
    
# Here, we generate the actor-critic network with relevant initialisations.
class Agent(object):
    def __init__(self, D_a, D_o, D_z, D_h, dt, num_of_agents, lr):
        self.D_a = D_a # The dimension of the action space of the agent
        self.D_o = D_o # The dimension of the observation space of the agent
        self.D_z = D_z # The dimension of the latent activity layer
        self.D_h = D_h # The dimension of the hidden activity layer
        self.N = num_of_agents # The number of agents in the environment
        self.dt = dt # Time step parameter
        self.sig_max = 2 # The maximum level of action noise
        self.sig_min = 0.01 # The minimum level of action noise
        self.VE_weight = 0.5 # Weighting given to the value-error term in the loss
        self.lbda = 0.98 # Decay parameter for the eligibility trace
        self.entropy_reg = 0.01 # Entropy regularisation of the policy
        self.MS_decay = 0.95 # Decay rate of the moving squared sum of the gradients
        self.lr = lr # Learning rate of the agent
        self.r_bar = 0 # Running average of the rewards, shared across all agents
        self.r_decay = 0.01 # Decay rate of the exponential recency weighted average of the rewards
        self.r_decay_norm = 0 # Normalisation for the average reward update
        self.v = torch.zeros(self.N, 1, device = DEVICE) # The estimated value of the current state
        self.v_prev = torch.zeros(self.N, 1, device = DEVICE) # The estimated value of the previous state
        self.delta = torch.zeros(self.N, 1, device = DEVICE) # The reward prediction error
        
        self.mu = torch.zeros(self.N, D_a, device = DEVICE) # Expected action
        self.sigma = torch.ones(self.N, 1, device = DEVICE) # Action noise
        
        self.z = torch.zeros(self.N, D_z, device = DEVICE) # Latent activities
        self.h = torch.zeros(self.N, D_h, device = DEVICE) # Hidden activities
        
        self.Layers = nn.ModuleDict({
            "Observation": nn.Linear(D_z + D_o, D_h, device = DEVICE),
            "Hidden-Latent": nn.Linear(D_h + D_a, D_z, device = DEVICE),
            "Action": nn.Linear(D_z, D_a, device = DEVICE),
            "Noise": nn.Linear(D_z, 1, device = DEVICE),
            "Value": nn.Linear(D_z, 1, device = DEVICE)
        })
        
        self.Eligibility_traces = [{}]*self.N # This is a list of dictionaries containing the eligibility traces for each parameter 
        self.Entropy_gradients = {}  # This is a dictionary containing the entropy gradients for each parameter
        self.Total_gradients = {} # This is a dictionary containing the total gradients for each parameter
        self.MS = {} # This is a dictionary containing the moving squared sum of the gradients for each parameter
        
        for name, param in self.Layers.named_parameters():
            self.Total_gradients[name] = torch.zeros_like(param, device = DEVICE, requires_grad = False) 
            self.Entropy_gradients[name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)
            self.MS[name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)
            for kk in range(self.N):
                self.Eligibility_traces[kk][name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)

    def load_weights(self, path):
        # This method loads the weights from a given path
        self.Layers.load_state_dict(torch.load(path))
            
    def sample_action(self):
        mu_unnorm = self.Layers["Action"](self.z) # mu_unnorm has dimensions (N, D_a)
        self.mu = F.normalize(mu_unnorm, p = 2, dim = 1) # normalize the action to have unit norm for each agent. mu has dimensions (N, D_a)
        self.sigma = (self.sig_max - self.sig_min)*F.sigmoid(self.Layers["Noise"](self.z)) + self.sig_min # sigma has dimensions (N, 1)
        a = self.mu + self.sigma*torch.randn(self.N, self.D_a, device = DEVICE) # a has dimensions (N, D_a)
        return a.detach() 
    
    def compute_value(self):
        self.v = self.Layers["Value"](self.z) # v has dimensions (N, 1)
        
    def compute_delta(self, r):
        # Compute the reward prediction error 
        self.update_r_bar(r) # Update the running average of the rewards
        self.v_prev = self.v.clone().detach() # Update the previous value estimate
        self.compute_value() # Compute the current value estimate
        self.delta = r.unsqueeze(1) - self.r_bar*torch.ones(self.N,1,device=DEVICE) + self.v.clone().detach() - self.v_prev # Compute the reward prediction error. This has dimensions (N, 1)
    
    def update_activities(self, a, o):
        self.h = F.relu(self.Layers["Observation"]( torch.cat((self.z, o), dim = 1) )) 
        self.z = (1 - self.dt)*self.z + self.dt*F.tanh(self.Layers["Hidden-Latent"]( torch.cat((self.h, a), dim = 1) ))
        
    def update_r_bar(self, r):
        self.r_decay_norm = self.r_decay_norm + self.r_decay*(1 - self.r_decay_norm) # Compute the normalisation constant for exponential recency weighted averaging
        self.r_bar = r.mean() + (self.r_decay/self.r_decay_norm)*( r.mean() - self.r_bar ) # Update the exponential recency weighted average
        
    def update_ET(self, a):
        # This finds the gradients of the log-policy and the value function.

        action_loss = -(torch.linalg.vector_norm(a - self.mu, dim = 1, keepdim = True)**2)/(2*self.sigma*self.sigma) - self.D_a*torch.log(self.sigma) # This has dimensions (N, 1)
        value_loss = self.VE_weight*self.v # This has dimensions (N, 1)
        loss = action_loss + value_loss # This has dimensions (N, 1)
        
        for kk in range(self.N):
            for name, param in self.Layers.named_parameters():
                # param.grad = torch.zeros_like(param, device = DEVICE) # Zero out the gradients
                param.grad = None
            loss[kk].backward(retain_graph = True)
            nn.utils.clip_grad_value_(self.Layers.parameters(), clip_value=1) # Clip the gradients to have absolute value at most 1
            for name, param in self.Layers.named_parameters(): 
                if param.grad is not None:
                    self.Eligibility_traces[kk][name] = self.lbda*self.Eligibility_traces[kk][name] + param.grad
                
    def recompute_outputs(self):
        # When run, this method simply recomputes the value estimate, expected action, and action noise from the current latent activities. 
        self.compute_value()
        mu_unnorm = self.Layers["Action"](self.z) # mu_unnorm has dimensions (N, D_a)
        self.mu = F.normalize(mu_unnorm, p = 2, dim = 1) # normalize the action to have unit norm for each agent. mu has dimensions (N, D_a)
        self.sigma = (self.sig_max - self.sig_min)*F.sigmoid(self.Layers["Noise"](self.z)) + self.sig_min # sigma has dimensions (N, 1)
    
    def update_EntGrad(self):
        # Here we find the gradients with respect to the entropy. 
        
        # First we zero out all the gradients:
        for name, param in self.Layers.named_parameters():
            # param.grad = torch.zeros_like(param, device = DEVICE)
            param.grad = None

        # Now we define the function of which we wish to take the derivative:
        Entropy = self.D_a*torch.log(self.sigma).mean() 
        Entropy.backward(retain_graph = True)
        nn.utils.clip_grad_value_(self.Layers.parameters(), clip_value=1) # Clip the gradients to have absolute value at most 1
        
        for name, param in self.Layers.named_parameters():
            if param.grad is not None:
                self.Entropy_gradients[name] = param.grad
            
    def update_total_gradient(self):
        for name, _ in self.Layers.named_parameters():
            self.Total_gradients[name] = self.Total_gradients[name] + self.entropy_reg*self.Entropy_gradients[name]
            for kk in range(self.N):
                self.Total_gradients[name] = self.Total_gradients[name] + (1/self.N)*self.delta[kk]*self.Eligibility_traces[kk][name]
                        
    def turn(self, o, r, a):
        # This function takes in O_{t+1}, R_{t+1} and A_t
        self.update_ET(a) # We compute the eligibility traces for time t 
        self.update_activities(a, o) # Now we calculate z_{t+1}
        self.compute_delta(r) # We compute \delta_t = R_{t+1} - bar{R}_{t+1} + V(z_{t+1}) - V(z_t)
        self.update_total_gradient() # We compute g_t = \delta_t ET_t + grad H[ pi(z_t) ]
        next_a = self.sample_action() # We sample action A_{t+1} 
        self.update_EntGrad() # We calculate the entropy gradient for policy pi(z_{t+1})
        return next_a # We return the action A_{t+1}
    
    def reset(self):
        # Reset all the variables of the network
        self.z = torch.zeros(self.N, self.D_z, device = DEVICE) # Latent activities
        self.h = torch.zeros(self.N, self.D_h, device = DEVICE) # Hidden activities
        self.v = torch.zeros(self.N, 1, device = DEVICE) # Value estimates
        self.v_prev = torch.zeros(self.N, 1, device = DEVICE) # Value estimate from the previous time step
        self.delta = torch.zeros(self.N, 1, device = DEVICE) # TD errors

        self.mu = torch.zeros(self.N, self.D_a, device = DEVICE) # Expected action
        self.sigma = torch.zeros(self.N, device = DEVICE) # Action noise

        # We zero out the gradients
        for name, param in self.Layers.named_parameters():
            param.grad = None
        
        # We also zero out the eligibility traces and the entropy gradients
        for name, param in self.Layers.named_parameters():
            self.Entropy_gradients[name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)
            for kk in range(self.N):
                self.Eligibility_traces[kk][name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)

    def sever(self):
        # This function prevents gradients flowing backwards and zeros out the eligibility trace.
        #self.z = self.z.clone().detach()
        self.z.detach_()
        self.h.detach_()
        self.v.detach_()
        self.v_prev.detach_()
        self.delta.detach_()
        self.mu.detach_()
        self.sigma.detach_()

        # We zero out the gradients
        for name, param in self.Layers.named_parameters():
            param.grad = None
        
        self.recompute_outputs()
        
        for name, param in self.Layers.named_parameters():
            self.Total_gradients[name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)
            self.Entropy_gradients[name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)
            for kk in range(self.N):
                self.Eligibility_traces[kk][name] = torch.zeros_like(param, device = DEVICE, requires_grad = False)

        
    
    def update_weights(self, num_of_steps):
        # This method updates the weights of the network using the total gradient
        for name, param in self.Layers.named_parameters():
            # First, average the total gradients over the number of steps
            self.Total_gradients[name] = self.Total_gradients[name]/num_of_steps
            # Now update the mean square estimate
            self.MS[name] = self.MS_decay*self.MS[name] + (1 - self.MS_decay)*self.Total_gradients[name]**2
            # Now perform the RMSprop update
            param.data = param.data + self.lr*self.Total_gradients[name]/(torch.sqrt(self.MS[name]) + 1e-8)