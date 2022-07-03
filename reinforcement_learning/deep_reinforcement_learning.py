"""
Deep Reinforcement Learning

This is the implementation of Deep Learning algorithm with Reinforcement Learning 
for the Half Cheetah problem.

"""



from turtle import shape
import gym
import numpy as np
from sklearn.model_selection import learning_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action,-1., 1.)

class Hyperparameters():

    def __init__(self) -> None:
        
        self.env_name = 'HalfCheetah-v4'   # Evironment Name
        self.limit = 1000                  # Memory limit
        self.window_len = 1                # Memory window length
        self.theta = .15                   # Ornstein-Uhlenbeck process parameter
        self.mu = 0.                       # Ornstein-Uhlenbeck process parameter mean
        self.sigma = .1                    # Ornstein-Uhlenbeck process parameter std
        self.nb_steps_warmup_critic = 1000 # Number of steps before update the Critic network
        self.nb_steps_warmup_actor  = 1000 # Number of steps before update the Actor network
        self.gamma = 0.99                  # Parameter to compute the discounted reward
        self.target_model_update = 1e-3    # DDPGA parameter



class Actor():

    """ 
    Actor Class

    This class implements the deep learning model that maps the
    state to an action.
    
    """

    def __init__(self, n_actions, n_neurons, n_layers, observarion_space) -> None:
        
        super().__init__()
        self.obs_space = observarion_space
        self.n_neurons = n_neurons
        self.n_layers = n_layers


        self.model = Sequential()
        self.model.add(Flatten(
                input_shape=(1,) + self.obs_space, name=f"actor_flatten_layer"
            ))
        for i in range(self.n_layers):
            self.model.add(
                Dense(n_neurons, name=f"actor_dense_layer_{i}")
            )
            self.model.add(
                Activation('relu', name=f"actor_activation_{i}")
            )
            n_neurons = int(n_neurons/2)
            if n_neurons <= n_actions:
                break
        
        self.model.add(
            Dense(n_actions, name=f"actor_final_dense_layer")
        )
        self.model.add(
            Activation('linear', name=f"actor_output_layer")
        )


class Critic():

    def __init__(self, action_input, obs_input, n_actions, obs_space, n_neurons, n_layers) -> None:
        super().__init__()

        self.n_actions = n_actions
        self.obs_space = obs_space
        self.n_layers = n_layers


        flattened_obs = Flatten(name="flatten_layer")(obs_input)
        x = Concatenate(name="concatenate_layer")([flattened_obs, action_input])
        for i in range(self.n_layers):
            x = Dense(n_neurons, name=f"critic_dense_layer_{i}")(x)
            x = Activation('relu', name=f"activation_layer_{i}")(x)
            n_neurons = int(n_neurons/2)
            if n_neurons <= 1:
                break
        
        x = Dense(1, name=f"critic_final_dense_layer")(x)
        x = Activation('linear', name=f"critic_output_layer")(x)
        self.model = Model(inputs=[action_input,obs_input], outputs=x)


class RLFramerowk():

    def __init__(self) -> None:

        self.hp = Hyperparameters()
        self.env = gym.make(self.hp.env_name)
        self.n_actions = self.env.action_space.shape[0]
        self.obs_space = self.env.observation_space.shape
        
        self.actor_neurons = 600
        self.actor_layers = 3
        self.critic_neurons = 800
        self.critic_layers = 5

        self.action_input = Input(shape=(self.n_actions,),name="action_input")
        self.obs_input = Input(shape=(1,)+self.obs_space)

        self.actor = Actor(
            self.n_actions,
            self.actor_neurons,
            self.actor_layers,
            self.obs_space
        )

        self.critic = Critic(
            self.action_input,
            self.obs_input,
            self.n_actions,
            self.obs_space,
            self.critic_neurons,
            self.critic_layers
        )

       
        self.memory = SequentialMemory(limit=self.hp.limit, window_length=self.hp.window_len)
        self.random_process = OrnsteinUhlenbeckProcess(
            size = self.n_actions,
            theta = self.hp.theta,
            mu = self.hp.mu,
            sigma = self.hp.sigma
        )

        self.agent = DDPGAgent(
            nb_actions = self.n_actions, actor = self.actor.model, critic = self.critic.model,
            critic_action_input = self.action_input, memory = self.memory,
            nb_steps_warmup_critic=self.hp.nb_steps_warmup_critic,
            nb_steps_warmup_actor=self.hp.nb_steps_warmup_actor,
            random_process=self.random_process,gamma=self.hp.gamma,
            target_model_update=self.hp.target_model_update
        )

        self.agent.compile(
            [
                Adam(learning_rate=1e-4),
                Adam(learning_rate=1e-3),
            ], metrics=['mae']
        )

        
    def learn(self, steps: int, visualize: bool = True, verbose: int = 1):

        self.agent.fit(
            env=self.env,
            nb_steps=steps,
            visualize=visualize,
            verbose=verbose
        )

        self.agent.save_weights(f'./reinforcement_learning/weights/ddpg_agent_{self.hp.env_name}_weights.h5f')

    def test(self, episodes: int=5, max_episodes:int=1000, visualize: bool=True):

        self.agent.load_weights(f'./reinforcement_learning/weights/ddpg_agent_{self.hp.env_name}_weights.h5f')
        self.agent.test(
            self.env,
            nb_episodes=episodes,
            visualize=visualize,
            nb_max_episode_steps=max_episodes
        )