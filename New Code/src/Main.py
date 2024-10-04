# ---------------------------------------------------------------------
#                               Main.py
#                    University of Central Florida
#              Autonomous & Intelligent Systems Labratory
#
# Description: Please run the simulation from here to test our model.
#
#
# Author        Date        Description
# ------        ----        ------------
# Jaxon Topel   10/3/24     Initial Architecture Creation
# ---------------------------------------------------------------------

# Import necessary modules.
from Environment import Environment
from ModelTrain import train_agent
from NeuralNetwork import DQN

# Define hyperparameters (adjust these as needed)
# -----------------------------------------------
# Controls how much the model's weights are updated in each iteration of training.
# Affects how quickly the agent learns from its experiences.
learning_rate = 0.001
# Number of training iterations.
num_episodes = 1000
# Number of testing iterations.
num_evaluation_episodes = 10

# Initialize the simulation environment for Simultaneous Localization and mapping (SLAM).
env = Environment()

# Define and create the neural network.
model = DQN(env.observation_space.shape[0], env.action_space.n)  

# Train the agent.
agent = train_agent(model, env, learning_rate, num_episodes)

# Evaluate the trained agent.
evaluate_agent(agent, env, num_evaluation_episodes)

# Close the environment.
env.close()