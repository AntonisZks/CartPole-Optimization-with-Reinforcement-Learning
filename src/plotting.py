import matplotlib.pyplot as plt
import numpy as np


plots_save_path = 'reports/figs/'
plt.style.use('default')


def plot_default_scores(scores: list[float]) -> None:
    """
    Plots the scores obtained from running the environment with random actions.

    Args:
        scores (list[float]): List of scores obtained in each epoch.
    """

    # Create a figure and axis for the plot, set titles and labels
    figure, axis = plt.subplots(1, 1, figsize=(7, 5))
    axis.plot(scores, color='dodgerblue')
    axis.set_ylim(0, 550)
    axis.set_title("Environement Random Action testing: Total Score per Epoch\n")
    axis.set_xlabel("Epochs")
    axis.set_ylabel("Score")
    axis.grid(color='grey')

    # Adjust layout, save the figure, and display it
    plt.tight_layout()
    plt.savefig(plots_save_path + 'dummy_scores.png')
    plt.show()


def plot_training_results(results: dict, training_title: str = "Training Score", file_name: str = "results") -> None:
    """
    Plots the training results including scores and epsilon decrease over epochs.
    It creates a figure with two subplots: one for scores and another for epsilon values.

    Args:
        results (dict): Dictionary containing 'scores' and 'epsilons' lists.
        training_title (str, optional): Title for the training score plot. Defaults to "Training Score".
        file_name (str, optional): Filename to save the plot. Defaults to "results".
    """    
    
    # Create a figure with two subplots for scores and epsilon values
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].plot(results['scores'], color='dodgerblue')
    axis[0].set_ylim(0, 550)
    axis[0].set_title(training_title)
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Score")
    axis[0].grid(color='grey')

    # Plot epsilon values on the second subplot, set titles and labels
    axis[1].plot(results['epsilons'], color='orange')
    axis[1].set_title("Epsilon Decrease")
    axis[1].set_xlabel("Epochs")
    axis[1].set_ylabel("Epsilon")
    axis[1].set_ylim(0, 1.1)
    axis[1].grid(color='grey')

    # Adjust layout, save the figure, and display it
    plt.tight_layout()
    plt.savefig(plots_save_path + f'{file_name}.png')
    plt.show()


def plot_trained_vs_dummy_results(
    dummy_scores: list[float], dummy_wins: int, dummy_fails: int, 
    test_epochs_scores: list[float], trained_wins: int, trained_fails: int) -> None:
    '''
    Plots a comparison between the performance of a DQN agent and random actions in the CartPole-v1 environment.
    It creates a figure with two subplots: one for scores and another for wins and fails.
    
    Args:
        dummy_scores (list[float]): Scores obtained from random actions.
        dummy_wins (int): Number of wins from random actions.
        dummy_fails (int): Number of fails from random actions.
        test_epochs_scores (list[float]): Scores obtained from the DQN agent.
        trained_wins (int): Number of wins from the DQN agent.
        trained_fails (int): Number of fails from the DQN agent.
    '''
    
    # Comparing the results between random actions and DQN agent
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].plot(dummy_scores, label='Random Actions', color='#FF4646')
    axis[0].plot(test_epochs_scores, label='DQN Agent', color='#2DBB23')

    axis[0].set_ylim(0, 550)
    axis[0].set_title("DQN Agent vs Random Actions: Total Score per Epoch")
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Score")
    axis[0].legend()
    axis[0].grid(color='grey')

    categories = ['Random Actions', 'DQN Agent']
    x = np.arange(len(categories))  # the label locations
    width = 0.25  # the width of the bars
    gap = 0.02

    wins_values = [dummy_wins, trained_wins]
    fails_values = [dummy_fails, trained_fails]

    bars_wins = axis[1].bar(x - width/2 - gap, wins_values, width, label='Wins', color=["#2DBB23", '#2DBB23'])
    bars_fails = axis[1].bar(x + width/2 + gap, fails_values, width, label='Fails', color=['#FF4646', '#FF4646'])

    axis[1].set_title("Number of Wins and Fails Comparison")
    axis[1].set_ylabel("Number of Episodes")
    axis[1].set_xticks(x)
    axis[1].set_xticklabels(categories)
    axis[1].legend()

    # Adding labels on top of the bars
    for bar in bars_wins:
        height = bar.get_height()
        axis[1].text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom')

    for bar in bars_fails:
        height = bar.get_height()
        axis[1].text(bar.get_x() + bar.get_width() / 2.0, height, f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plots_save_path + 'dqn_vs_random.png')
    plt.show()

    print(f'Number of wins with DQN Agent: {trained_wins}')
    print(f'Number of fails with DQN Agent: {trained_fails}')