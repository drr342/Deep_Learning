
# coding: utf-8

# ## 0. Setup

# In[1]:


# Import dependencies
import torch
import torch.nn as nn
from plot_lib import plot_data
from matplotlib.pyplot import plot, title, axis
from matplotlib import pyplot as plt


# In[2]:


# Set up your device 
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")


# In[3]:


# Set up random seed to 1008. Do not change the random seed.
seed = 1008
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)


# ## 1. Data generation
# #### You'll be creating data points that are generated from a particular function.

# ### 1.1 Quadratic: $y = f(x) = x^2$

# In[4]:


# Implement the function below
def quadratic_data_generator(n_samples):
    """
    Generate: 
    1) tensor x of size (n_samples, 1) 
    with values uniformly distributed in the interval (-1, 1] 
    using torch.rand()
    2) tensor y of size (n_samples, 1) 
    equal to x^2 using torch.pow() 
    
    The function should return: x, y
    """
    x = torch.add(torch.mul(torch.rand(n_samples), -2), 1)
    y = torch.pow(x, 2)
    return x.to(device), y.to(device)


# In[5]:


# Generate the data with n_samples = 128
x_quadr, y_quadr = quadratic_data_generator(128)


# In[6]:


# Visualize the data
plot_data(x_quadr, y_quadr, auto=True, title="f(x) = x^2")


# ### 1.2 Cubic: $y = f(x) = x^3 - 0.5x$

# In[7]:


# Implement the function below
def cubic_data_generator(n_samples):
    """
    Generate: 
    1) tensor x of size (n_samples, 1) 
    with values uniformly distributed in the interval (-1, 1] 
    using torch.rand()
    2) tensor y of size (n_samples, 1) 
    equal to (x^3 - 0.5x) using torch.pow() and torch.mul() 
    
    The function should return: x, y
    """
    x = torch.add(torch.mul(torch.rand(n_samples), -2), 1)
    y = torch.pow(x, 3) - torch.mul(x, 0.5)
    return x.to(device), y.to(device)


# In[8]:


# Generate the data with n_samples = 128
x_cubic, y_cubic = cubic_data_generator(128)


# In[9]:


# Visualize the data 
plot_data(x_cubic, y_cubic, auto=True, title="f(x) = x^3-0.5x")


# ### 1.3 Sine: $y = f(x) = \sin(2.5x)$

# In[10]:


# Implement the function below
def sine_data_generator(n_samples):
    """
    Generate: 
    1) tensor x of size (n_samples, 1) 
    with values uniformly distributed in the interval (-1, 1] 
    using torch.rand()
    2) tensor y of size (n_samples, 1) 
    equal to sin(2.5 * x) using torch.sin() 
    
    The function should return: x, y
    """
    x = torch.add(torch.mul(torch.rand(n_samples), -2), 1)
    y = torch.sin(torch.mul(x, 2.5))
    return x.to(device), y.to(device)


# In[11]:


# Generate the data with n_samples = 128
x_sine, y_sine = sine_data_generator(128)


# In[12]:


# Visualize the data 
plot_data(x_sine, y_sine, auto=True, title="f(x) = sin(2.5x)")


# ### 1.4 Absolute value: $y = f(x) = |x|$

# In[13]:


# Implement the function below
def abs_data_generator(n_samples):
    """
    Generate: 
    1) tensor x of size (n_samples, 1) 
    with values uniformly distributed in the interval (-1, 1] 
    using torch.rand()
    2) tensor y of size (n_samples, 1) 
    equal to |x| using torch.abs() 
    
    The function should return: x, y
    """
    x = torch.add(torch.mul(torch.rand(n_samples), -2), 1)
    y = torch.abs(x)
    return x.to(device), y.to(device)


# In[14]:


# Generate the data with n_samples = 128
x_abs, y_abs = abs_data_generator(128)


# In[15]:


# Visualize the data 
plot_data(x_abs, y_abs, auto=True, title="f(x) = |x|")


# ### 1.5 Heavyside Step Function: $y = f(x) = \begin{cases} 0, & x < 0 \\ 1, & x \geq 0 \end{cases}$

# In[16]:


# Implement the function below
def hs_data_generator(n_samples):
    """
    Generate: 
    1) tensor x of size (n_samples, 1) 
    with values uniformly distributed in the interval (-1, 1] 
    using torch.rand()
    2) tensor y of size (n_samples, 1) 
    equal to the Heavyside Step Function using a condition.
    Make sure that y is a torch.FloatTensor.
    
    The function should return: x, y
    """
    x = torch.add(torch.mul(torch.rand(n_samples), -2), 1)
    y = (x >= 0).type(torch.FloatTensor)
    return x.to(device), y.to(device)


# In[17]:


# Generate the data with n_samples = 128
x_hs, y_hs = hs_data_generator(128)


# In[18]:


# Visualize the data 
plot_data(x_hs, y_hs, auto=True, title="f(x) = H(x)")


# ## 2. Models
# #### You are going to approximate the functions above with fully connected models of different depths.  

# ### 2.1. Dimensionality
# The models you define below will be predicting $y$ from $x$ and will use the data generated in Part 1 as training data. Fill in the input and output dimensions for each of the models.
# 
# Hint: These dimensions are independent from the number of samples. 

# In[19]:


input_dim = 1
output_dim = 1


# ### 2.2. No Hidden 
# Define a model with a single linear module `torch.nn.Linear(input_dim, output_dim)` and no non-linearity.

# In[20]:


class Linear_0H(nn.Module):
    def __init__(self):
        super(Linear_0H, self).__init__()
        
        # Layers
        self.network = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.network(x)


# ### 2.2. One Hidden 
# Define a model with a single hidden layer of size 3 and one ReLU non-linearity.
# Use `nn.Sequential()` for defining the layers.
# 
# Hint: Architecture should be `nn.Linear(intput_dim, 3)` -> `nn.ReLU()` -> `nn.Linear(3, output_dim)`

# In[21]:


class Linear_1H(nn.Module):
    def __init__(self):
        super(Linear_1H, self).__init__()
        self.n_hidden = 3
        
        # Layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.n_hidden), 
            nn.ReLU(), 
            nn.Linear(self.n_hidden, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ### 2.3. Two Hidden 
# Define a model with a two hidden layers of size 3 and two ReLU non-linearities.
# Use `nn.Sequential()` for defining the layers.
# 
# Hint: Architecture should be `nn.Linear(input_dim,3)` -> `nn.ReLU()` -> `nn.Linear(3,3)` -> `nn.ReLU()` -> `nn.Linear(3, output_dim)`

# In[22]:


class Linear_2H(nn.Module):
    def __init__(self):
        super(Linear_2H, self).__init__()
        self.n_hidden = 3
        
        # Layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.n_hidden), 
            nn.ReLU(), 
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(), 
            nn.Linear(self.n_hidden, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ## 3. Training

# ### 3.1 Train method
# You are going to implement a training method which takes a model, number of epochs, training data, and threshold for loss functions as input and returns the (detached) predicitons from the last epoch. 
# 
# Make sure you understand what the method is doing and how early stopping works in this case.

# In[23]:


# Training function
def train(model, epochs, x, y, loss_threshold=1e-2):
    # Set model to training mode
    model.train()
    
    # Define Mean Squared Error as loss function using nn.MSELoss()
    criterion = nn.MSELoss()
    
    # Define the SGD optimizer with learning rate of 0.01 using torch.optim.SGD()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    # Define if the model finished training early
    early_stop = False
    
    # Training loop
    for epoch in range(epochs):
        # Forward data through model 
        y_pred = model(x.view(-1, 1))
        
        # Compute the loss 
        loss = criterion(y_pred, y.view(-1, 1))
        
        # Zero-out the optimizer 
        optimizer.zero_grad()
        
        # Backpropagate loss
        loss.backward()
        
        # Make a step with the optimizer
        optimizer.step()
        
        # Uncomment lines below once you implement the code above
        # Print out loss every 100 epochs 
        if epoch == 0 or (epoch+1) % 1000 == 0:
            print('Epoch {} loss: {}'.format(epoch+1, loss.item()))
        
        # Uncomment lines below once you implement the code above
        # Early stopping based on training loss
        if loss.item() < loss_threshold:
            print('Epoch {} loss: {}'.format(epoch+1, loss.item()))
            early_stop = True
            break    
    
    # Return predictions from the last epoch.
    # Uncomment line below once you implement
    return y_pred.detach(), early_stop


# ### 3.2. `Linear_0H`

# In[24]:


# Define list of booleans for early stop
early_stop = [False for _ in range(15)]


# In[25]:


# Define model
model_0H = Linear_0H().to(device)


# In[26]:


# Train model on quadratic data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[0] = train(model_0H, epochs=10000, x=x_quadr, y=y_quadr, loss_threshold=1e-2)


# In[27]:


# Plot predictions vs actual data
plot_data(y_quadr, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_quadr, y_pred.view(-1))))
plot_data(x_quadr, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[28]:


# Define model
model_0H = Linear_0H().to(device)


# In[29]:


# Train model on cubic data for 10000 epochs and loss_threshold=1e-2
y_pred, early_stop[1] = train(model_0H, epochs=10000, x=x_cubic, y=y_cubic, loss_threshold=1e-2)


# In[30]:


# Plot predictions vs actual data
plot_data(y_cubic, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_cubic, y_pred.view(-1))))
plot_data(x_cubic, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[31]:


# Define model
model_0H = Linear_0H().to(device)


# In[32]:


# Train model on sine data for 10000 epochs and loss_threshold=1e-2
y_pred, early_stop[2] = train(model_0H, epochs=10000, x=x_sine, y=y_sine, loss_threshold=1e-2)


# In[33]:


# Plot predictions vs actual data
plot_data(y_sine, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_sine, y_pred.view(-1))))
plot_data(x_sine, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[34]:


# Define model
model_0H = Linear_0H().to(device)


# In[35]:


# Train model on abosulte value data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[3] = train(model_0H, epochs=10000, x=x_abs, y=y_abs, loss_threshold=1e-2)


# In[36]:


# Plot predictions vs actual data
plot_data(y_abs, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_abs, y_pred.view(-1))))
plot_data(x_abs, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[37]:


# Define model
model_0H = Linear_0H().to(device)


# In[38]:


# Train model on Heavyside Step Function data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[4] = train(model_0H, epochs=10000, x=x_hs, y=y_hs, loss_threshold=1e-2)


# In[39]:


# Plot predictions vs actual data
plot_data(y_hs, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_hs, y_pred.view(-1))))
plot_data(x_hs, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# ### 3.3. `Linear_1H`

# In[40]:


# Define model
model_1H = Linear_1H().to(device)


# In[41]:


# Train model on quadratic data for 10000 epochs and loss threshold 1e-2
y_pred, early_stop[5] = train(model_1H, epochs=10000, x=x_quadr, y=y_quadr, loss_threshold=1e-2)


# In[42]:


# Plot predictions vs actual data
plot_data(y_quadr, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_quadr, y_pred.view(-1))))
plot_data(x_quadr, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[43]:


# Define model
model_1H = Linear_1H().to(device)


# In[44]:


# Train model on cubic data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[6] = train(model_1H, epochs=10000, x=x_cubic, y=y_cubic, loss_threshold=1e-2)


# In[45]:


# Plot predictions vs actual data
plot_data(y_cubic, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_cubic, y_pred.view(-1))))
plot_data(x_cubic, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[46]:


# Define model
model_1H = Linear_1H().to(device)


# In[47]:


# Train model on sine data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[7] = train(model_1H, epochs=10000, x=x_sine, y=y_sine, loss_threshold=1e-2)


# In[48]:


# Plot predictions vs actual data
plot_data(y_sine, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_sine, y_pred.view(-1))))
plot_data(x_sine, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[49]:


# Define model
model_1H = Linear_1H().to(device)


# In[50]:


# Train model on abosulte value data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[8] = train(model_1H, epochs=10000, x=x_abs, y=y_abs, loss_threshold=1e-2)


# In[51]:


# Plot predictions vs actual data
plot_data(y_abs, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_abs, y_pred.view(-1))))
plot_data(x_abs, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[52]:


# Define model
model_1H = Linear_1H().to(device)


# In[53]:


# Train model on Heavyside Step Function data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[9] = train(model_1H, epochs=10000, x=x_hs, y=y_hs, loss_threshold=1e-2)


# In[54]:


# Plot predictions vs actual data
plot_data(y_hs, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_hs, y_pred.view(-1))))
plot_data(x_hs, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# ### 3.3. `Linear_2H`

# In[55]:


# Define model
model_2H = Linear_2H().to(device)


# In[56]:


# Train model on quadratic data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[10] = train(model_2H, epochs=10000, x=x_quadr, y=y_quadr, loss_threshold=1e-2)


# In[57]:


# Plot predictions vs actual data
plot_data(y_quadr, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_quadr, y_pred.view(-1))))
plot_data(x_quadr, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[58]:


# Define model
model_2H = Linear_2H().to(device)


# In[59]:


# Train model on cubic data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[11] = train(model_2H, epochs=10000, x=x_cubic, y=y_cubic, loss_threshold=1e-2)


# In[60]:


# Plot predictions vs actual data
plot_data(y_cubic, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_cubic, y_pred.view(-1))))
plot_data(x_cubic, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[61]:


# Define model
model_2H = Linear_2H().to(device)


# In[62]:


# Train model on sine data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[12] = train(model_2H, epochs=10000, x=x_sine, y=y_sine, loss_threshold=1e-2)


# In[63]:


# Plot predictions vs actual data
plot_data(y_sine, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_sine, y_pred.view(-1))))
plot_data(x_sine, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[64]:


# Define model
model_2H = Linear_2H().to(device)


# In[65]:


# Train model on abosulte value data for 10000 epochs and loss_threshold=1e-2
y_pred, early_stop[13] = train(model_2H, epochs=10000, x=x_abs, y=y_abs, loss_threshold=1e-2)


# In[66]:


# Plot predictions vs actual data
plot_data(y_abs, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_abs, y_pred.view(-1))))
plot_data(x_abs, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[67]:


# Define model
model_2H = Linear_2H().to(device)


# In[68]:


# Train model on Heavyside Step Function data for 10000 epochs and loss_threshold 1e-2
y_pred, early_stop[14] = train(model_2H, epochs=10000, x=x_hs, y=y_hs, loss_threshold=1e-2)


# In[69]:


# Plot predictions vs actual data
plot_data(y_hs, y_pred.view(-1), auto=True, title=None)

# Plot input vs predictions and actual data
plt.figure()
data = torch.t(torch.stack((y_hs, y_pred.view(-1))))
plot_data(x_hs, data, color=['r', 'b'], auto=True, legend=['Actual', 'Predicted'])


# In[70]:


# Print list of models that stopped early
print(early_stop)


# ### 3.4. Which of the models stopped early and on what data?
# Please list the experiments where the `loss_threshold` of 1e-2 was reached early. 

# #### List: 
#     Linear_1H - Cubic function
#     Linear_1H - Absolut value
#     Linear_2H - Cubic function
#     Linear_2H - Sine function
