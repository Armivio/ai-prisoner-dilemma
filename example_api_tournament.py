"""
Example of running a Prisoner's Dilemma tournament with real AI models.

Before running:
1. Copy .env.example to .env
2. Add your API keys to .env
3. Install dependencies: pip install -r requirements.txt
"""

from prisoners_dilemma_tournament import PrisonersDilemmaTournament
from api_models import test_api_connectivity
import os
from dotenv import load_dotenv


def run_ai_tournament():
    """Run a tournament with real AI models."""
    
    # Load environment variables
    load_dotenv()
    
    # First, test API connectivity
    print("=" * 60)
    test_api_connectivity()
    print("=" * 60)
    print()
    
    # Define the AI models to compete
    # You can use any combination of models from different providers
    models = ["claude-3-5-haiku-20241022", "claude-3-7-sonnet-20250219", "claude-sonnet-4-20250514", "claude-opus-4-20250514", "gpt-4o-mini"]
    if not models:
        print("No API keys found! Please add at least one API key to your .env file.")
        print("Example:")
        print("  ANTHROPIC_API_KEY=your-key-here")
        print("  OPENAI_API_KEY=your-key-here")
        print("  GOOGLE_API_KEY=your-key-here")
        return
    
    print(f"Running tournament with {len(models)} AI models:")
    for model in models:
        print(f"  - {model}")
    print()
    
    # Create tournament with fewer games to save API costs
    tournament = PrisonersDilemmaTournament(
        models=models,
        games_per_matchup=5,  # Reduced from 100 to save API calls
        enable_cross_memory=True  # Enable memory across opponents
    )
    
    print("Starting tournament... (this may take a few minutes)")
    print("Each dot represents a completed game:")
    print()
    
    # Run the tournament
    results = tournament.run_tournament()
    
    # Display results
    print("\n\n=== TOURNAMENT RESULTS ===\n")
    
    print("Final Leaderboard:")
    print("-" * 50)
    for rank, (player, score) in enumerate(results.get_leaderboard(), 1):
        avg_score = score / (len(models) - 1) / tournament.games_per_matchup
        print(f"{rank}. {player:<20} {score:>5} points (avg: {avg_score:.2f}/game)")
    
    # Show cooperation rates
    print("\n\nCooperation Rates:")
    print("-" * 50)
    stats = results.get_summary_stats()
    for player, rate in sorted(stats['cooperation_rate'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{player:<20} {rate:>6.1%}")
    
    # Show head-to-head summary
    print("\n\nHead-to-Head Win Records:")
    print("-" * 50)
    matrix = results.get_head_to_head_matrix()
    
    for player in sorted(models):
        wins = sum(matrix[player][opp]["wins"] for opp in models if opp != player)
        losses = sum(matrix[player][opp]["losses"] for opp in models if opp != player)
        ties = sum(matrix[player][opp]["ties"] for opp in models if opp != player)
        print(f"{player:<20} Wins: {wins:>3}, Losses: {losses:>3}, Ties: {ties:>3}")
    
    # Show a sample game between top two players
    if len(results.get_leaderboard()) >= 2:
        top_player = results.get_leaderboard()[0][0]
        second_player = results.get_leaderboard()[1][0]
        
        print(f"\n\nSample games between top two players:")
        print(f"{top_player} vs {second_player}")
        print("-" * 50)
        
        history = results.get_matchup_history(top_player, second_player)
        if history and history.games:
            for game in history.games[:5]:  # Show first 5 games
                p1_move = "Cooperate" if game.player1_move == 'c' else "Defect"
                p2_move = "Cooperate" if game.player2_move == 'c' else "Defect"
                print(f"Round {game.round_num}: "
                      f"{game.player1}={p1_move}, {game.player2}={p2_move} "
                      f"(scores: {game.player1_score}, {game.player2_score})")


def run_minimal_test():
    """Run a minimal test with just 2 games to verify setup."""
    load_dotenv()
    
    # Find available models
    models = []
    if os.getenv("ANTHROPIC_API_KEY"):
        models.append("claude-3-5-haiku-20241022")
    if os.getenv("OPENAI_API_KEY"):
        models.append("gpt-4o-mini")
    # if os.getenv("GOOGLE_API_KEY"):
        # models.append("gemini-flash")
    
    if len(models) < 2:
        print("Need at least 2 different API keys for a tournament!")
        return
    
    # Use just the first 2 models
    models = models[:2]
    
    print(f"Running minimal test with: {models[0]} vs {models[1]}")
    print("Playing just 2 games to test the setup...\n")
    
    tournament = PrisonersDilemmaTournament(
        models=models,
        games_per_matchup=2,  # Just 2 games
        enable_cross_memory=False
    )
    
    results = tournament.run_tournament()
    
    # Show the results
    history = results.get_matchup_history(models[0], models[1])
    if history:
        for game in history.games:
            print(f"Round {game.round_num}: "
                  f"{game.player1}={game.player1_move.upper()}, "
                  f"{game.player2}={game.player2_move.upper()} "
                  f"(scores: {game.player1_score}, {game.player2_score})")
    
    print(f"\nFinal scores: {models[0]}={results.total_scores[models[0]]}, "
          f"{models[1]}={results.total_scores[models[1]]}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run minimal test
        run_minimal_test()
    else:
        # Run full tournament
        run_ai_tournament()