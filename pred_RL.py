# file_path: streamlit_app_rl.py

import streamlit as st
import pandas as pd
import numpy as np
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import matplotlib.pyplot as plt

# Streamlit configuration
st.set_page_config(page_title="Meal Order Prediction with RL", layout="centered")

# Define custom environment for meal prediction
class MealPredictionEnv(gym.Env):
    def __init__(self, num_meals=10):
        super(MealPredictionEnv, self).__init__()
        self.num_meals = num_meals
        self.action_space = spaces.Discrete(num_meals)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.state = np.random.rand(5)
        return self.state

    def step(self, action):
        preference_score = np.dot(self.state, np.random.rand(5))
        reward = 1 if action == np.argmax(preference_score) else -1
        self.state = np.random.rand(5)
        done = False
        return self.state, reward, done, {}

    def render(self):
        print(f'User Preferences: {self.state}')

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Streamlit app for training and using RL model
def main():
    st.title("Meal Order Prediction using Reinforcement Learning")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Choose a CSV file:", type="csv")

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Preprocess and setup environment
        env = MealPredictionEnv(num_meals=10)
        agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

        # Training the RL agent
        episodes = 500
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])

            for time in range(200):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [1, agent.state_size])
                agent.train(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

        st.success("RL Model trained successfully!")

        # User Input for prediction
        st.header("Enter Details to Predict Orders")
        user_input = np.random.rand(1, env.observation_space.shape[0])  # Example random input

        if st.button("Predict Orders"):
            action = agent.act(user_input)
            st.write(f"Recommended Meal ID: {action}")

if __name__ == "__main__":
    main()
