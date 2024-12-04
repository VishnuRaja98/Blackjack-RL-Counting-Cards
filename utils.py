import numpy as np
from collections import defaultdict
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle


# Random Agent
class RandomAgent:
    def get_action(self, state):
        return np.random.choice([0, 1])
    

# Define the default Q-value function
def default_q_value():
    return np.zeros(2)  # Two actions: 0 (Stand), 1 (Hit)

# Q-Learning Agent
class QLearningAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9999,
        min_epsilon=0.01
    ):
        self.q_values = defaultdict(default_q_value)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(self.q_values[state])
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action

    def update(self, state, action, reward, next_state, done):
        current_q = self.q_values[state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_values[next_state])
        self.q_values[state][action] += self.lr * (target_q - current_q)

# SARSA Agent
class SARSAAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.9999,
        min_epsilon=0.01
    ):
        self.q_values = defaultdict(default_q_value)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice([0, 1])
        else:
            action = np.argmax(self.q_values[state])
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return action

    def update(self, state, action, reward, next_state, next_action, done):
        current_q = self.q_values[state][action]
        if done:
            target_q = reward
        else:
            # Use the Q-value of the next state-action pair for updating
            target_q = reward + self.gamma * self.q_values[next_state][next_action]
        
        # Update the Q-value for the current state-action pair
        self.q_values[state][action] += self.lr * (target_q - current_q)


# Blackjack Environment
class BlackjackEnv(gym.Env):
    def __init__(self, render_mode=None, natural=False, sab=False, total_decks=5, include_count=False):
        super(BlackjackEnv, self).__init__()
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
        if BlackjackEnv.usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    @staticmethod
    def is_bust(hand):
        return BlackjackEnv.sum_hand(hand) > 21

    @staticmethod
    def score(hand):
        return 0 if BlackjackEnv.is_bust(hand) else BlackjackEnv.sum_hand(hand)

    @staticmethod
    def is_natural(hand):
        return sorted(hand) == [1, 10]