from collections import defaultdict
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import gymnasium as gym
from gymnasium import spaces


class AgentRandom:
    def get_action(self, env) -> int:
        return env.action_space.sample()


class AgentQL:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
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
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs: tuple[int, int, bool, int, float]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool, int, float],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool, int, float],
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


class AgentSARSA:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
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
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs: tuple[int, int, bool, int, float]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon, return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon), act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool, int, float],
        action: int,
        reward: float,
        next_obs: tuple[int, int, bool, int, float],
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


class NN(torch.nn.Module):
    def __init__(self, include_count):
        input_dim = 5 if include_count else 3
        self.linear1 = torch.nn.Linear(input_dim, 24)
        self.linear2 = torch.nn.Linear(24, 24)
        self.linear3 = torch.nn.Linear(24, 2)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        return self.softmax(output)
    

class AgentDQL:
    def __init__(
        self,
        env,
        learning_rate: float,
        discount_factor: float = 0.95,
        include_count = True
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and a NN.

        Args:
            learning_rate: The learning ratenin
            discount_factor: The discount factor for computing the Q-value
        """

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.include_count = include_count
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.training_error = []
        self.model = NN(include_count)

    def get_action(self, env, obs: tuple[int, int, bool, int, float]) -> int:
        """
        Returns the action using neural network forward pass
        """
        # convert obs into a tensor
        if self.include_count:
            input = torch.tensor([obs[0], obs[1], obs[2], obs[3], int(obs[4] * 2)])
        else:
            input = torch.tensor([obs[0], obs[1], obs[2]])
        return int(self.model(input))

    def update(
        self,
        obs: tuple[int, int, bool, int, float],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool, int, float],
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
        

# Blackjack Environment
class BlackjackEnvironment(gym.Env):
    def __init__(self, render_mode=None, natural=False, sab=False, total_decks=5, include_count=False):
        super(BlackjackEnvironment, self).__init__()
        self.include_count = include_count
        self.action_space = gym.spaces.Discrete(2)  # 0: Stand, 1: Hit

        if self.include_count:
            self.observation_space = gym.spaces.Tuple(
                (
                    gym.spaces.Discrete(32),   # Player's current sum (0-31)
                    gym.spaces.Discrete(11),   # Dealer's visible card (1-10)
                    gym.spaces.Discrete(2),    # Usable Ace (0: No, 1: Yes)
                    gym.spaces.Discrete(101),  # Player's money (0-100)
                    gym.spaces.Discrete(41),   # Running Count shifted from [-20,20] to [0,40]
                    gym.spaces.Discrete(21)     # Remaining Decks (0-20)
                )
            )
        else:
            self.observation_space = gym.spaces.Tuple(
                (
                    gym.spaces.Discrete(32),   # Player's current sum (0-31)
                    gym.spaces.Discrete(11),   # Dealer's visible card (1-10)
                    gym.spaces.Discrete(2),    # Usable Ace (0: No, 1: Yes)
                    gym.spaces.Discrete(101)   # Player's money (0-100)
                )
            )

        self.natural = natural
        self.sab = sab
        self.render_mode = render_mode
        self.running_count = 0
        self.betting_unit = 1
        self.money = 50
        self.current_bet = 1
        one_suite = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        self.original_deck = one_suite * 4 * total_decks
        self.deck = self.original_deck.copy()

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid Action: {action}"
        if action == 1:
            self.player.append(self.draw_card())
            if self.is_bust(self.player):
                terminated = True
                reward = -1.0
            else:
                terminated = False
                reward = 0.0
        else:
            terminated = True
            while self.sum_hand(self.dealer) < 17:
                self.dealer.append(self.draw_card())
            reward = self.cmp(self.score(self.player), self.score(self.dealer))
            if self.sab and self.is_natural(self.player) and not self.is_natural(self.dealer):
                reward = 1.0
            elif not self.sab and self.natural and self.is_natural(self.player) and reward == 1.0:
                reward = 1.5

        # Update money and ensure it doesn't go below 0
        self.money = max(self.money + reward * self.current_bet, 0)
        return self._get_state(), reward, terminated
    def _get_obs(self):
        if self.include_count:
            shifted_count = self.running_count + 20
            shifted_count = np.clip(shifted_count, 0, 40)
            remaining_decks = self.getRemainingDecks()
            return (
                self.sum_hand(self.player),
                self.dealer[0],
                int(self.usable_ace(self.player)),
                self.money,
                shifted_count,
                remaining_decks
            )
        else:
            return (
                self.sum_hand(self.player),
                self.dealer[0],
                int(self.usable_ace(self.player)),
                self.money
            )
        
    def _get_state(self):
        return self.sum_hand(self.player), self.dealer[0], int(self.usable_ace(self.player)), self.running_count, self.getRemainingDecks()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.money = 50
        self.deck = self.original_deck.copy()
        self.running_count = 0
        return self.new_round()

    def new_round(self):
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        return self._get_state()

    def draw_card(self):
        if len(self.deck) == 0:
            raise ValueError("The deck is empty!")
        card = self.deck.pop(np.random.randint(len(self.deck)))
        if self.include_count:
            if card in [1, 10]:
                self.running_count -= 1
            elif 2 <= card <= 6:
                self.running_count += 1
        return card

    def draw_hand(self):
        return [self.draw_card(), self.draw_card()]

    def getRemainingDecks(self):
        return min(round(len(self.deck) / 52 * 2) / 2, 20)

    @staticmethod
    def cmp(a, b):
        return float(a > b) - float(a < b)

    @staticmethod
    def usable_ace(hand):
        return 1 in hand and sum(hand) + 10 <= 21

    @staticmethod
    def sum_hand(hand):
        if BlackjackEnvironment.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    @staticmethod
    def is_bust(hand):
        return BlackjackEnvironment.sum_hand(hand) > 21

    @staticmethod
    def score(hand):
        return 0 if BlackjackEnvironment.is_bust(hand) else BlackjackEnvironment.sum_hand(hand)

    @staticmethod
    def is_natural(hand):
        return sorted(hand) == [1, 10]


def plot_train_stats(env, agent):
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    axs[0].set_title("Episode rewards")
    # compute and assign a rolling average of the data to provide a smoother graph
    reward_moving_average = (
        np.convolve(
            np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[1].set_title("Episode lengths")
    length_moving_average = (
        np.convolve(
            np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
        )
        / rolling_length
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()