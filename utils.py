from collections import defaultdict
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class BlackjackObservationWrapper(gym.Wrapper):
    def __init__(self, env, num_decks=1):
        super().__init__(env)
        self.num_decks = num_decks
        self.total_cards = num_decks * 52
        self.cards_drawn = 0
        self.running_count = 0
        
        # Update observation space to include running_count and num_decks_remaining
        original_obs_space = self.env.observation_space
        self.observation_space = spaces.Tuple((
            original_obs_space[0],  # Player's current sum
            original_obs_space[1],  # Dealer's one showing card
            original_obs_space[2],  # Usable ace
            spaces.Discrete(200),  # Running count (assume a reasonable range)
            spaces.Discrete(num_decks + 1)  # Number of decks remaining
        ))
    
    def reset(self, **kwargs):
        # Reset the environment and custom attributes
        self.cards_drawn = 0
        self.running_count = 0
        
        # Get the base observation
        observation, info = self.env.reset(**kwargs)
        
        # Calculate the additional state values
        num_decks_remaining = self._calculate_decks_remaining()
        return (*observation, self.running_count, num_decks_remaining), info

    def step(self, action):
        # Take a step in the base environment
        observation, reward, done, truncated, info = self.env.step(action)
        
        # Update running count and cards drawn
        self._update_running_count_and_drawn(observation)
        
        # Calculate the additional state values
        num_decks_remaining = self._calculate_decks_remaining()
        modified_observation = (*observation, self.running_count, num_decks_remaining)
        return modified_observation, reward, done, truncated, info

    def _update_running_count_and_drawn(self, observation):
        """Updates the running count using Hi-Lo strategy and tracks cards drawn."""
        player_sum, dealer_card, _ = observation
        
        # Update the count for dealer's showing card
        self.running_count += self._hi_lo_value(dealer_card)
        self.cards_drawn += 1

        # For simplicity, assume each player action draws 1 card
        self.running_count += self._hi_lo_value(player_sum)
        self.cards_drawn += 1

    def _hi_lo_value(self, card):
        """Return the Hi-Lo count value for a given card."""
        if 2 <= card <= 6:
            return 1
        elif card == 1 or card >= 10:  # Ace or face cards
            return -1
        return 0

    def _calculate_decks_remaining(self):
        """Calculate the number of decks remaining based on cards drawn."""
        cards_left = self.total_cards - self.cards_drawn
        return max(0, cards_left // 52)  # Avoid negative decks remaining


class BlackjackAgentQLearning:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        true_random: bool = False
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.true_random = true_random
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs: tuple[int, int, bool, int, int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if self.true_random or np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool, int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool, int, int],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class BlackjackAgentSARSA:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        true_random: bool = False,
    ):
        """
        Initialize a SARSA agent with an empty dictionary
        of state-action values (q_values), a learning rate, and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.true_random = true_random
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs: tuple[int, int, bool, int, int]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon, return a random action to explore the environment
        if self.true_random or np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon), act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool, int, int],
        action: int,
        reward: float,
        next_obs: tuple[int, int, bool, int, int],
        next_action: int,
        terminated: bool,
    ):
        """Updates the Q-value of an action using the SARSA update formula."""
        future_q_value = (not terminated) * self.q_values[next_obs][next_action]
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
