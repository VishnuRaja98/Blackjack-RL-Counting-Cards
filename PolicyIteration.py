import random
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Plot function for visualizing the overall reward
def plot_lines(data):
    plt.figure(figsize=(10, 6))
    differences = [lst[-1] - lst[0] for lst in data]
    min_diff = np.min(differences)
    max_diff = np.max(differences)
    median_diff = np.median(differences)
    std_diff = np.std(differences)

    norm = plt.Normalize(min(differences), max(differences))
    colors = plt.cm.plasma(norm(differences))

    for idx, values in enumerate(data):
        plt.plot(values, color=colors[idx])

    stats_text = (
        f"Most Loss: {min_diff:.2f}\n"
        f"Most Profit: {max_diff}\n"
        f"Median Profit: {median_diff}\n"
        f"Std Dev: {std_diff:.2f}"
    )
    plt.text(0.05, 0.95, stats_text,
         transform=plt.gca().transAxes,
         fontsize=10,
         verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.5))

    plt.title('Line Graphs of ' + str(len(data)) + ' episodes')
    plt.xlabel('Rounds Played')
    plt.ylabel('Money')
    plt.grid(True)
    plt.savefig("blackjack_policy_iteration_returns.png", dpi=300, bbox_inches='tight')


class State:
    def __init__(self, player_sum, dealer_hand, running_count):
        self.player_sum = player_sum
        self.dealer_hand = dealer_hand
        self.running_count = running_count

    def __eq__(self, other):
        if isinstance(other, State):
            return (self.player_sum == other.player_sum and
                    self.dealer_hand == other.dealer_hand and
                    self.running_count == other.running_count)
        return False

    def __hash__(self):
        return hash((self.player_sum, self.dealer_hand, self.running_count))

    def __str__(self):
        return f"[Player Sum: {self.player_sum}, Dealer Hand: {self.dealer_hand}, Running Count: {self.running_count}]"


class BlackjackGame:
    def __init__(self, start_money, num_decks=5):
        self.start_money = start_money
        self.current_money = start_money
        self.num_decks = num_decks
        self.deck = self.create_deck(num_decks)
        self.money_history = [start_money]
        self.running_count = 0
        self.history = []  # To store [s, a, r] tuples
        self.bet_values = [25, 50, 75, 100]
        self.bet_probs = [0.65, 0.2, 0.1, 0.05]
        self.play_values = ["hit", "stand"]

    def create_deck(self, n=1):
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        deck = [(rank, suit) for suit in suits for rank in ranks] * n
        random.shuffle(deck)
        return deck

    def deal_card(self, update_count=False):
        new_card = self.deck.pop()
        if update_count:
            self.update_count(new_card)
        return new_card

    def update_count(self, card):
        rank = card[0]
        if rank in ["2", "3", "4", "5", "6"]:
            self.running_count += 1
        elif rank in ["10", "J", "Q", "K", "A"]:
            self.running_count -= 1

    def calculate_value(self, hand):
        value = 0
        num_aces = 0
        for card in hand:
            rank = card[0]
            if rank in ["J", "Q", "K"]:
                value += 10
            elif rank == "A":
                num_aces += 1
                value += 11
            else:
                value += int(rank)

        while value > 21 and num_aces > 0:
            value -= 10
            num_aces -= 1

        return value

    def get_bet_amount(self):
        bet_amount = random.choices(self.bet_values, weights=self.bet_probs)[0]
        while bet_amount > self.current_money:
            bet_amount = random.choices(self.bet_values, weights=self.bet_probs)[0]
        return bet_amount

    def play_round(self, policy):
        player_hand = []
        dealer_hand = []
        bet_amount = self.get_bet_amount()

        # Deal initial cards
        player_hand.append(self.deal_card(update_count=True))
        dealer_hand.append(self.deal_card(update_count=True))
        player_hand.append(self.deal_card(update_count=True))
        dealer_hand.append(self.deal_card(update_count=False))

        # Create initial state
        state = State(self.calculate_value(player_hand), self.calculate_value([dealer_hand[0]]), self.running_count)

        if self.calculate_value(player_hand) == 21:  # player has blackjack
            if self.calculate_value(dealer_hand) == 21:  # dealer also has blackjack
                reward = 0
            else:
                reward = bet_amount * 1.5
            self.update_count(dealer_hand[1])
            next_state = State(self.calculate_value(player_hand), self.calculate_value(dealer_hand), self.running_count)
            self.history.append((state, "stand", reward, next_state))
            self.current_money += reward
        else:
            action = policy[state]  # Choose action based on policy
            while action == "hit":
                player_hand.append(self.deal_card())
                self.update_count(player_hand[-1])

                if self.calculate_value(player_hand) > 21:
                    reward = -bet_amount
                    next_state = State(self.calculate_value(player_hand), self.calculate_value(dealer_hand), self.running_count)
                    self.history.append((state, action, reward, next_state))
                    self.current_money += reward
                    break
                else:
                    reward = 0
                    next_state = State(self.calculate_value(player_hand), self.calculate_value([dealer_hand[0]]), self.running_count)
                    self.history.append((state, action, reward, next_state))
                    action = policy[next_state]  # Get next action after hit
            else:  # stand
                while self.calculate_value(dealer_hand) < 17:
                    dealer_hand.append(self.deal_card())
                    self.update_count(dealer_hand[-1])

                player_value = self.calculate_value(player_hand)
                dealer_value = self.calculate_value(dealer_hand)

                if dealer_value > 21 or player_value > dealer_value:
                    reward = bet_amount
                elif dealer_value > player_value:
                    reward = -bet_amount
                else:
                    reward = 0

                next_state = State(player_value, dealer_value, self.running_count)
                self.history.append((state, action, reward, next_state))  # action is stand
                self.current_money += reward

        self.money_history.append(self.current_money)
        return reward  # Return the reward for the round

    def play_game(self, policy):
        while self.current_money > 0 and len(self.deck) > 26:
            self.play_round(policy)
        return self.money_history, self.history

    def reset(self):
        self.current_money = self.start_money
        self.deck = self.create_deck(self.num_decks)
        self.money_history = [self.start_money]
        self.history = []


def initialize_policy():
    policy = {}
    for player_sum in range(4, 22):  # Player sums range from 4 to 21
        for dealer_card in range(1, 11):  # Dealer upcard ranges from 1 to 10
            state = State(player_sum, dealer_card, 0)  # Assuming running_count starts at 0
            # Initialize the policy randomly with 'hit' or 'stand' actions
            policy[state] = random.choice(["hit", "stand"])
    return policy


# Policy Evaluation Function
def policy_evaluation(policy, game, num_episodes=100):
    V = {}
    for _ in range(num_episodes):
        game.reset()
        total_reward = 0
        while game.current_money > 0:
            # Calculate the initial state before any rounds are played
            state = State(game.calculate_value([]), game.calculate_value([game.deal_card(update_count=False)]), game.running_count)
            
            # Access the policy as a dictionary instead of calling it as a function
            action = policy[state]  # Policy should map states to actions
            reward = game.play_round(action)
            total_reward += reward
        V[state] = total_reward / num_episodes
    return V


# Policy Improvement Function
def policy_improvement(game, V):
    new_policy = {}
    for state in V:
        best_action = None
        max_value = -float('inf')
        for action in game.play_values:
            reward = 0  # Calculate expected reward for action
            next_state = State(game.calculate_value(state), game.calculate_value([action]), game.running_count)
            value = reward + V.get(next_state, 0)
            if value > max_value:
                max_value = value
                best_action = action
        new_policy[state] = best_action
    return new_policy


# Policy Iteration Function
def policy_iteration(game, num_episodes=100):
    policy = initialize_policy()  # Initialize the policy with random actions
    
    V = policy_evaluation(policy, game, num_episodes)
    while True:
        new_policy = policy_improvement(game, V)
        if new_policy == policy:
            break
        policy = new_policy
        V = policy_evaluation(policy, game, num_episodes)
    return policy


if __name__ == "__main__":
    # Example run command:
    # python PolicyIteration.py --num_episodes 100 --start_money 200

    parser = argparse.ArgumentParser(description="Policy Iteration for Blackjack.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes for policy iteration.")
    parser.add_argument("--start_money", type=int, default=200, help="Starting money for the player.")
    args = parser.parse_args()

    start_money = args.start_money
    game = BlackjackGame(start_money)
    policy = policy_iteration(game, args.num_episodes)

    # Run the simulation with the learned policy
    money_histories = []
    for _ in range(args.num_episodes):
        game.reset()
        money_history, _ = game.play_game(policy)
        money_histories.append(money_history)

    plot_lines(money_histories)
