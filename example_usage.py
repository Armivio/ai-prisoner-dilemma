from prisoners_dilemma_tournament import PrisonersDilemmaTournament, get_prediction
from typing import Dict, Any
import random


# Example implementation of get_prediction for testing
# In real usage, this would interface with actual AI models
def mock_get_prediction(model_input: Dict[str, Any]) -> str:
    """
    Mock implementation of get_prediction for demonstration.
    In production, this would call actual AI models.
    """
    model_name = model_input["model"]
    history = model_input.get("history", "")
    
    # Simple strategies for demonstration
    if "always-cooperate" in model_name:
        return "c"
    elif "always-defect" in model_name:
        return "d"
    elif "tit-for-tat" in model_name:
        # Cooperate first, then copy opponent's last move
        if "Round 1:" not in history:
            return "c"
        else:
            # Extract opponent's last move from history
            lines = history.strip().split('\n')
            if lines:
                last_line = lines[-1]
                if "Opponent played D" in last_line:
                    return "d"
                else:
                    return "c"
            return "c"
    elif "random" in model_name:
        return random.choice(["c", "d"])
    else:
        # Default strategy
        return "c"


# Override the get_prediction function for this example
import prisoners_dilemma_tournament
prisoners_dilemma_tournament.get_prediction = mock_get_prediction


def run_example_tournament():
    """Run an example tournament with various strategies"""
    
    # Define some example models with different strategies
    models = [
        "always-cooperate-bot",
        "always-defect-bot",
        "tit-for-tat-bot",
        "random-bot-1",
        "random-bot-2"
    ]
    
    print("Setting up Prisoner's Dilemma Tournament")
    print(f"Players: {', '.join(models)}")
    print(f"Games per matchup: 20")
    print("-" * 60)
    
    # Create and run tournament
    tournament = PrisonersDilemmaTournament(
        models=models,
        games_per_matchup=20,
        enable_cross_memory=False
    )
    
    results = tournament.run_tournament()
    
    # Display results
    print("\n=== TOURNAMENT RESULTS ===\n")
    
    print("Final Leaderboard:")
    print("-" * 40)
    for rank, (player, score) in enumerate(results.get_leaderboard(), 1):
        print(f"{rank}. {player:<25} {score:>5} points")
    
    print("\n\nHead-to-Head Results:")
    print("-" * 60)
    matrix = results.get_head_to_head_matrix()
    
    # Print header
    print(f"{'Player':<25}", end="")
    for player in sorted(models):
        print(f"{player[:8]:<10}", end="")
    print()
    
    # Print matrix
    for player1 in sorted(models):
        print(f"{player1:<25}", end="")
        for player2 in sorted(models):
            if player1 == player2:
                print(f"{'---':^10}", end="")
            else:
                stats = matrix[player1][player2]
                print(f"{stats['total_score']:^10}", end="")
        print()
    
    print("\n\nDetailed Statistics:")
    print("-" * 40)
    stats = results.get_summary_stats()
    print(f"Total games played: {stats['total_games']}")
    
    print("\nAverage score per game:")
    for player, avg in sorted(stats['average_score_per_game'].items()):
        print(f"  {player:<25} {avg:>6.2f}")
    
    print("\nCooperation rate:")
    for player, rate in sorted(stats['cooperation_rate'].items()):
        print(f"  {player:<25} {rate:>6.1%}")
    
    # Show a sample game history
    print("\n\nSample Game History (first matchup):")
    print("-" * 40)
    first_matchup = list(results.matchup_results.values())[0]
    print(f"{first_matchup.player1} vs {first_matchup.player2}")
    for game in first_matchup.games[:5]:  # Show first 5 games
        print(f"  Round {game.round_num}: "
              f"{game.player1}={game.player1_move.upper()}, "
              f"{game.player2}={game.player2_move.upper()} "
              f"(scores: {game.player1_score}, {game.player2_score})")


def demonstrate_cross_memory():
    """Demonstrate tournament with cross-opponent memory enabled"""
    print("\n\n=== TOURNAMENT WITH CROSS-MEMORY ENABLED ===\n")
    
    models = ["adaptive-bot-1", "adaptive-bot-2", "adaptive-bot-3"]
    
    tournament = PrisonersDilemmaTournament(
        models=models,
        games_per_matchup=10,
        enable_cross_memory=True
    )
    
    # For demonstration, let's trace what inputs are prepared
    print("Example of model input with cross-memory:")
    print("-" * 60)
    
    # Simulate some history
    tournament.all_history["adaptive-bot-1"]["adaptive-bot-2"] = [("c", "c"), ("c", "d"), ("d", "d")]
    
    # Prepare an example input
    example_input = tournament.prepare_model_input(
        model="adaptive-bot-1",
        opponent="adaptive-bot-3",
        round_num=5,
        history=[("c", "c"), ("c", "d"), ("d", "c"), ("d", "d")]
    )
    
    print("Model receives:")
    print(f"- Current opponent: {example_input['opponent']}")
    print(f"- Current round: {example_input['round']}")
    print(f"- History with current opponent: {len(example_input['history'].split('Round')) - 1} games")
    if example_input['cross_memory']:
        print(f"- Cross-memory data: Available")
    print("\nFull prompt preview:")
    print(example_input['prompt'][:500] + "...")


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run example tournament
    run_example_tournament()
    
    # Demonstrate cross-memory feature
    demonstrate_cross_memory()