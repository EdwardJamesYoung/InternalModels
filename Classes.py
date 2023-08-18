import numpy as np
from torch import nn
import torch.nn.functional as F
import copy
import torch
torch.autograd.set_detect_anomaly(True)

# Basic admin
np.random.seed(seed = 1234)

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE = "cpu"

# Build the environment class
class Environment(object):
    def __init__(self, D_s, D_o, D_a, dt, clone = False):
        self.D_s = D_s # Dimension of the latent environmental state
        self.D_o = D_o # Dimension of the observation space
        self.D_a = D_a # Dimension of the action space
        self.dt = dt # Time-step parameter
        
        if not clone: 
            # We randomly generate the dynamics of the environment 
            self.Q = dt*np.sqrt(2/(D_s + D_a))*torch.randn(D_s, D_a, device = DEVICE)
            self.K = np.sqrt(2/(D_s + D_o))*torch.randn(D_o, D_s, device = DEVICE)
            self.rho = 1/np.sqrt(D_s)*torch.randn(D_s, device = DEVICE)
            
            # Sample P such that the leading eigenvalue has absolute value below 1
            eigV = 1.1*torch.ones(D_s)
            while not (eigV <= 1).all():
                self.P = (1 - dt)*torch.eye(D_s, device = DEVICE) + (dt/np.sqrt(D_s))*torch.randn(D_s, D_s, device = DEVICE)
                eigV, _ = torch.linalg.eig(self.P)
                eigV = abs(eigV)
        else: 
            self.Q = torch.zeros(D_s,D_a, device = DEVICE)
            self.K = torch.zeros(D_o, D_s, device = DEVICE)
            self.rho = torch.zeros(D_s, device = DEVICE)
            self.P = torch.zeros(D_s, D_s, device = DEVICE)
            
        # Define the state vector
        self.s = torch.zeros(D_s, device = DEVICE)
    
    def step(self, a):
        # Given an action, update the state vector.
        self.s = self.P@self.s + self.Q@a
        self.s.detach()
        
        # Return the observation of the next state, along with the reward associated with that state. 
        o = (self.K@self.s).detach()
        r = torch.dot(self.rho,self.s).detach()
        return o, r
    
    def reset(self):
        self.s = (0.4/np.sqrt(self.D_s))*torch.randn(self.D_s, device = DEVICE)
        
    def clone_environment(self):
        env_copy = Environment(self.D_s,self.D_o,self.D_a,self.dt)
        env_copy.Q = self.Q
        env_copy.K = self.K
        env_copy.rho = self.rho
        env_copy.P = self.P
        
        return env_copy
    
# Here, we generate the actor-critic network with relevant initialisations.
class Agent (object):
    def __init__(self, D_a, D_o, D_z, D_h, dt):
        self.D_a = D_a # The dimension of the action space of the agent
        self.D_o = D_o # The dimension of the observation space of the agent
        self.D_z = D_z # The dimension of the latent activity layer
        self.D_h = D_h # The dimension of the hidden activity layer
        self.dt = dt # Time step parameter
        self.sig_max = 1 # The maximum level of action noise
        self.epsilon = 0.001 # Additive constant to prevent division by zero in action sampling
        self.VE_weight = 1 # Weighting given to the value-error term in the loss
        self.lbda = 0.99 # Decay parameter for the eligibility trace
        self.entropy_reg = 0.01 # Entropy regularisation of the policy
        self.r_bar = 0 # Running average of the rewards
        self.r_decay = 0.01 # Decay rate of the exponential recency weighted average of the rewards
        self.r_decay_norm = 0 # Normalisation for the average reward update
        self.v = 0 # The estimated value of the current state
        self.v_prev = 0 # The estimated value of the previous state
        self.delta = 0 # The reward prediction error
        
        self.mu = torch.zeros(D_a, device = DEVICE) # Expected action
        self.sigma = 1 # Action noise
        
        self.z = torch.zeros(D_z, device = DEVICE) # Latent activities
        self.h = torch.zeros(D_h, device = DEVICE) # Hidden activities
        
        self.Layers = nn.ModuleDict({
            "Observation": nn.Linear(D_z + D_o, D_h),
            "Hidden-Latent": nn.Linear(D_h + D_a, D_z),
            "Action": nn.Linear(D_z, D_a),
            "Noise": nn.Linear(D_z, 1),
            "Value": nn.Linear(D_z, 1)
        })

        self.Layers.to(DEVICE)
        
        self.Eligibility_traces = {}
        self.Entropy_gradients = {}
        self.Total_gradients = {}
        
        for name, param in self.Layers.named_parameters():
            self.Eligibility_traces[name] = torch.zeros_like(param, device = DEVICE)
            self.Entropy_gradients[name] = torch.zeros_like(param, device = DEVICE)
            self.Total_gradients[name] = torch.zeros_like(param, device = DEVICE)
        
    def sample_action(self):
        mu_unnorm = self.Layers["Action"](self.z)
        self.mu = mu_unnorm/( self.epsilon + torch.linalg.vector_norm(mu_unnorm) ) 
        self.sigma = self.sig_max*F.sigmoid(self.Layers["Noise"](self.z))
        a = self.mu + self.sigma*torch.randn(self.D_a, device = DEVICE)
        a.detach()
        return a
    
    def compute_value(self):
        self.v = self.Layers["Value"](self.z)
        
    def compute_delta(self, r):
        # Compute the reward prediction error 
        self.update_r_bar(r)
        self.v_prev = self.v
        self.compute_value()
        self.delta = r - self.r_bar + self.v - self.v_prev 
    
    def update_activities(self, a, o):
        self.h = F.tanh(self.Layers["Observation"]( torch.cat((self.z, o)) ))
        self.z = self.z + self.dt*F.relu(self.Layers["Hidden-Latent"]( torch.cat((self.h, a)) ))
        
    def update_r_bar(self, r):
        self.r_decay_norm = self.r_decay_norm + self.r_decay*(1 - self.r_decay_norm) # Compute the normalisation constant for exponential recency weighted averaging
        self.r_bar = r + (self.r_decay/self.r_decay_norm)*( r - self.r_bar ) # Update the exponential recency weighted average
        
    def update_ET(self, a):
        # This finds the gradients of the log-policy and the value function.
        
        # First we zero out all the gradients:
        for name, param in self.Layers.named_parameters():
            param.grad = torch.zeros_like(param, device = DEVICE)

        # Now we define the function of which we wish to take the derivative:
        loss = - (torch.linalg.vector_norm(a - self.mu)**2)/(2*self.sigma*self.sigma) - self.D_a*torch.log(self.sigma) + self.VE_weight*self.v
        loss.backward(retain_graph = True)
        
        for name, params in self.Layers.named_parameters():
            self.Eligibility_traces[name] = self.lbda*self.Eligibility_traces[name] + params.grad
            
    def recompute_outputs(self):
        # When run, this method simply recomputes the value estimate, expected action, and action noise from the current latent activities. 
        self.compute_value()
        mu_unnorm = self.Layers["Action"](self.z)
        self.mu = mu_unnorm/( self.epsilon + torch.linalg.vector_norm(mu_unnorm) ) 
        self.sigma = self.sig_max*F.sigmoid(self.Layers["Noise"](self.z))
    
    def update_EntGrad(self):
        # Here we find the gradients with respect to the entropy. 
        
        # First we zero out all the gradients:
        for name, param in self.Layers.named_parameters():
            param.grad = torch.zeros_like(param, device = DEVICE)
            
        # Now we define the function of which we wish to take the derivative:
        Entropy = self.D_a*torch.log(self.sigma)
        Entropy.backward(retain_graph = True)
        
        for name, param in self.Layers.named_parameters():
            self.Entropy_gradients[name] = param.grad
            
    def update_total_gradient(self):
        for name, _ in self.Layers.named_parameters():
            self.Total_gradients[name] = self.Total_gradients[name] + self.delta*self.Eligibility_traces[name] + self.entropy_reg*self.Entropy_gradients[name]
    
    def copy_weights_to_agent(self, duplicate_agent):
        # This function takes in another agent, duplicate_agent, and loads the parameters of this agent into duplicate agent.
        duplicate_agent.Layers = copy.deepcopy(self.Layers)
            
    def clone_agent(self):
        agent_clone = Agent(self.D_a, self.D_o, self.D_z, self.D_h, self.dt)
        self.copy_weights_to_agent(agent_clone)
        return agent_clone
            
    def turn(self, o, r, a):
        # This function takes in O_{t+1}, R_{t+1} and A_t
        self.update_ET(a) # We compute ET_t
        self.update_activities(a, o) # Now we calculate z_{t+1}
        self.compute_delta(r) # We compute \delta_t = R_{t+1} - bar{R}_{t+1} + V(z_{t+1}) - V(z_t)
        self.update_total_gradient() # We compute g_t = \delta_t ET_t + grad H[ pi(z_t) ]
        a = self.sample_action() # We sample action A_{t+1} 
        self.update_EntGrad() # We calculate the entropy gradient for policy pi(z_{t+1})
        return a # We return the action A_{t+1}
    
    def reset(self):
        # Reset all the variables of the network
        self.z = torch.zeros(self.D_z, device = DEVICE) # Latent activities
        self.h = torch.zeros(self.D_h, device = DEVICE) # Hidden activities
        self.v = 0
        self.v_prev = 0 # The estimated value of the previous state
        self.delta = 0 # The reward prediction error
        
        self.mu = torch.zeros(self.D_a, device = DEVICE) # Expected action
        self.sigma = 1 # Action noise
        
        # We also zero out the eligibility traces and the entropy gradients
        for name, param in self.Layers.named_parameters():
            self.Eligibility_traces[name] = torch.zeros_like(param, device = DEVICE)
            self.Entropy_gradients[name] = torch.zeros_like(param, device = DEVICE)
    
    def sever(self):
        # This function prevents gradients flowing backwards and zeros out the eligibility trace.
        self.z = self.z.clone().detach()
        self.recompute_outputs()
        
        for name, param in self.Layers.named_parameters():
            self.Eligibility_traces[name] = torch.zeros_like(param, device = DEVICE)
            self.Entropy_gradients[name] = torch.zeros_like(param, device = DEVICE)
            self.Total_gradients[name] = torch.zeros_like(param, device = DEVICE)

class Network_optimiser(object):
    def __init__(self, network, lr):
        # Network is a module dict object
        self.network = copy.deepcopy(network)
        self.MS_decay = 0.95
        self.lr = lr
        
        self.Gradient_step = {}
        self.MS = {}
        self.Update_step = {}
        for name, param in self.network.named_parameters():
            self.Gradient_step[name] = torch.zeros_like(param)
            self.MS[name] = torch.zeros_like(param)
            self.Update_step[name] = torch.zeros_like(param)
        
    def load_weights_from_agent(self, agent):
        # This method loads the weights of the given network into the network attribute
        self.network = copy.deepcopy(agent.Layers)
            
    def copy_weights_to_agent(self, agent):
        # This method loads the weights of the network attribute into the given network
        agent.Layers = copy.deepcopy(self.network)    
            
    def average_gradients(self, grads, steps):
        # This function takes in a list of gradients and the number of time steps the gradients are over. 
        # We then sum the gradients, average across agents and across time, and save them in Gradient_step
        
        for name, param in self.network.named_parameters():
            # Begin by zeroing out the gradient step.
            self.Gradient_step[name] = torch.zeros_like(param, device = DEVICE)
            
            # Next, sum over gradients
            for ii in range(len(grads)):
                self.Gradient_step[name] = self.Gradient_step[name] + grads[ii][name]
            
            # Finally, average
            self.Gradient_step[name] = self.Gradient_step[name]/( len(grads)*steps )
        
    def update(self):
        # This is to be called after the Gradient_step has been calculated. 
        # We update the MS parameter, then perform the update
        for name, param in self.network.named_parameters():
            # Update the RMS parameter
            self.MS[name] = self.MS_decay*self.MS[name] + ( 1 - self.MS_decay )*(self.Gradient_step[name]**2)
            # Perform the update
            param = param + self.lr*( self.Gradient_step[name] )/torch.sqrt( self.MS[name] + 0.00001) 

