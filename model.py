import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys
import traci

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a Feed-Forward Neural Network
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


# a property basically means that we can access the variable without the ()
    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim
    

class PPOAgent:
    '''
    PPO Agent using tensorflow 2.0
    '''
    def __init__(self, input_dim, output_dim, policy_network):
        # We don't pass the environment, as the environment is accessed through traci library

        # observation space
        self.input_dim = input_dim
        # action space
        self.output_dim = output_dim
        # the NN
        self.policy_network = policy_network # from the TrainModel class, the model built
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1600
        '''ALG STEP 1'''
        self.actor = policy_network(input_dim, output_dim)
        # we use 1 as out_dim because we want to output a single value representing the estimated value of the state
        self.critic = policy_network(input_dim, 1)

    def call(self, obs):
        '''
        Forward pass of the neural network. Uses ReLU activation for hidden layers and linear activation for output layer
        '''
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        activation1 = tf.nn.relu(self.layer1(obs))
        activation2 = tf.nn.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

    # def choose_action(self, state):
    #     state = np.reshape(state, (1, self.input_dim))
    #     action_probs = self.policy_network.predict(state)[0]
    #     action = np.random.choice(self.output_dim, p=action_probs)
    #     return action

    # def train_step(self, states, actions, advantages, returns, old_policy):
    #     with tf.GradientTape() as tape:
    #         action_probs = self.policy_network(states)
    #         chosen_action_probs = tf.reduce_sum(tf.one_hot(actions, self.output_dim) * action_probs, axis=1)
    #         old_action_probs = tf.reduce_sum(tf.one_hot(actions, self.output_dim) * old_policy, axis=1)
    #         ratio = tf.exp(tf.math.log(chosen_action_probs) - tf.math.log(old_action_probs))
    #         clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
    #         surrogate = -tf.minimum(ratio * advantages, clipped_ratio * advantages)
    #         policy_loss = tf.reduce_mean(surrogate)
    #         value_loss = tf.reduce_mean(tf.square(returns - self.policy_network(states)))
    #         total_loss = policy_loss + self.value_coef * value_loss

    #     gradients = tape.gradient(total_loss, self.policy_network.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
    #     return total_loss

    # def calculate_advantages(self):
    #     pass
    #     # Calculate advantages using collected experiences

    # def calculate_returns(self):
    #     pass
    #     # Calculate returns using collected rewards

    # def train_step(self):
        pass
        # Perform a single training step of PPO algorithm

    def learn(self, total_timesteps):
        currSteps = 0
        '''AlG STEP 2'''
        while currSteps < total_timesteps:
            pass

    def rollout(self):
        '''
        Generate single batch data by performing timesteps_per_batch timesteps
        '''
        batch_obs = []             # batch observations (# of timespseps, obs_dim)
        batch_acts = []            # batch actions (# of timesteps, act_dim)
        batch_log_probs = []       # log probs of each action (# of timesteps)
        batch_rews = []            # batch rewards (# of episodes, # timeseps per episode)
        batch_rtgs = []            # batch rewards-to-go (# of timesteps)
        batch_lens = []            # episodic lengths in batch (# of episodes)
        # num of timeseps in batch so far
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            # connect to SUMO and start the simulation
            traci.start(self._sumo_cmd) # obs = self.env.reset()
            print("Simulating...")
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ = self.env.step(action)
            
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)
        






# Inside the training loop:
# advantages = agent.calculate_advantages(...)
# returns = agent.calculate_returns(...)
# agent.train_step(...)