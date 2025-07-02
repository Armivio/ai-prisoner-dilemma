from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

# Import the API-based get_prediction function
try:
    from api_models import get_prediction
except ImportError:
    # If api_models is not available, use the placeholder
    def get_prediction(model_input: Dict[str, Any]) -> str:
        """
        This function is PROVIDED EXTERNALLY - do not implement.
        It takes a model input dictionary and returns "c" or "d"
        """
        raise NotImplementedError("get_prediction must be implemented externally")


@dataclass
class GameResult:
    """Represents the result of a single game"""
    player1: str
    player2: str
    player1_move: str
    player2_move: str
    player1_score: int
    player2_score: int
    round_num: int


@dataclass
class MatchupHistory:
    """Tracks history between two specific players"""
    player1: str
    player2: str
    games: List[GameResult] = field(default_factory=list)
    
    def get_moves_for_player(self, player: str) -> Tuple[List[str], List[str]]:
        """Get lists of (own_moves, opponent_moves) for a specific player"""
        own_moves = []
        opponent_moves = []
        
        for game in self.games:
            if game.player1 == player:
                own_moves.append(game.player1_move)
                opponent_moves.append(game.player2_move)
            else:
                own_moves.append(game.player2_move)
                opponent_moves.append(game.player1_move)
        
        return own_moves, opponent_moves


class TournamentResults:
    """Manages and reports tournament results"""
    
    def __init__(self):
        self.total_scores: Dict[str, int] = defaultdict(int)
        self.matchup_results: Dict[Tuple[str, str], MatchupHistory] = {}
        self.all_games: List[GameResult] = []
    
    def add_game_result(self, result: GameResult):
        """Add a single game result to the tournament"""
        self.all_games.append(result)
        self.total_scores[result.player1] += result.player1_score
        self.total_scores[result.player2] += result.player2_score
        
        # Store in matchup history
        key = tuple(sorted([result.player1, result.player2]))
        if key not in self.matchup_results:
            self.matchup_results[key] = MatchupHistory(result.player1, result.player2)
        self.matchup_results[key].games.append(result)
    
    def get_leaderboard(self) -> List[Tuple[str, int]]:
        """Get sorted leaderboard of players and scores"""
        return sorted(self.total_scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_head_to_head_matrix(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get head-to-head results matrix"""
        players = sorted(self.total_scores.keys())
        matrix = {p1: {p2: {"wins": 0, "losses": 0, "ties": 0, "total_score": 0} 
                      for p2 in players} for p1 in players}
        
        for matchup_key, history in self.matchup_results.items():
            for game in history.games:
                if game.player1_score > game.player2_score:
                    matrix[game.player1][game.player2]["wins"] += 1
                    matrix[game.player2][game.player1]["losses"] += 1
                elif game.player2_score > game.player1_score:
                    matrix[game.player1][game.player2]["losses"] += 1
                    matrix[game.player2][game.player1]["wins"] += 1
                else:
                    matrix[game.player1][game.player2]["ties"] += 1
                    matrix[game.player2][game.player1]["ties"] += 1
                
                matrix[game.player1][game.player2]["total_score"] += game.player1_score
                matrix[game.player2][game.player1]["total_score"] += game.player2_score
        
        return matrix
    
    def get_matchup_history(self, player1: str, player2: str) -> Optional[MatchupHistory]:
        """Get the history of games between two specific players"""
        key = tuple(sorted([player1, player2]))
        return self.matchup_results.get(key)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the tournament"""
        stats = {
            "total_games": len(self.all_games),
            "players": list(self.total_scores.keys()),
            "average_score_per_game": {},
            "cooperation_rate": {}
        }
        
        for player in self.total_scores.keys():
            games_played = sum(1 for g in self.all_games 
                             if g.player1 == player or g.player2 == player)
            if games_played > 0:
                stats["average_score_per_game"][player] = self.total_scores[player] / games_played
                
                cooperations = sum(1 for g in self.all_games 
                                 if (g.player1 == player and g.player1_move == 'c') or
                                    (g.player2 == player and g.player2_move == 'c'))
                stats["cooperation_rate"][player] = cooperations / games_played
        
        return stats


class PrisonersDilemmaTournament:
    """Main tournament manager for Prisoner's Dilemma competitions"""
    
    # Scoring matrix: (player1_move, player2_move) -> (player1_score, player2_score)
    SCORING_MATRIX = {
        ('c', 'c'): (3, 3),  # Both cooperate
        ('c', 'd'): (0, 5),  # Player1 cooperates, Player2 defects
        ('d', 'c'): (5, 0),  # Player1 defects, Player2 cooperates
        ('d', 'd'): (1, 1),  # Both defect
    }
    
    def __init__(self, models: List[Any], games_per_matchup: int = 100, 
                 enable_cross_memory: bool = False):
        """
        Initialize the tournament.
        
        Args:
            models: List of model objects/identifiers
            games_per_matchup: Number of games each pair plays
            enable_cross_memory: Whether AIs remember games against other opponents
        """
        self.models = models
        self.games_per_matchup = games_per_matchup
        self.enable_cross_memory = enable_cross_memory
        self.results = TournamentResults()
        self.all_history = defaultdict(lambda: defaultdict(list))  # player -> opponent -> history
    
    def run_tournament(self) -> TournamentResults:
        """Run the complete round-robin tournament"""
        # Generate all unique pairs for round-robin
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                model1, model2 = self.models[i], self.models[j]
                self._play_matchup(model1, model2)
        
        return self.results
    
    def _play_matchup(self, model1: Any, model2: Any):
        """Play all games between two models"""
        # Track history for this specific matchup
        history1 = []  # model1's moves and opponent's moves
        history2 = []  # model2's moves and opponent's moves
        
        for round_num in range(1, self.games_per_matchup + 1):
            result = self.play_game(model1, model2, round_num, history1, history2)
            
            # Update histories
            history1.append((result.player1_move, result.player2_move))
            history2.append((result.player2_move, result.player1_move))
            
            # Update cross-opponent memory if enabled
            if self.enable_cross_memory:
                self.all_history[str(model1)][str(model2)].append(
                    (result.player1_move, result.player2_move))
                self.all_history[str(model2)][str(model1)].append(
                    (result.player2_move, result.player1_move))
            
            # Record the result
            self.results.add_game_result(result)
    
    def play_game(self, model1: Any, model2: Any, round_num: int, 
                  history1: List[Tuple[str, str]], history2: List[Tuple[str, str]]) -> GameResult:
        """Play a single game between two models"""
        # Get moves from both models
        move1 = self._get_model_move(model1, model2, round_num, history1)
        move2 = self._get_model_move(model2, model1, round_num, history2)
        
        # Calculate scores
        scores = self.SCORING_MATRIX[(move1, move2)]
        
        return GameResult(
            player1=str(model1),
            player2=str(model2),
            player1_move=move1,
            player2_move=move2,
            player1_score=scores[0],
            player2_score=scores[1],
            round_num=round_num
        )
    
    def _get_model_move(self, model: Any, opponent: Any, round_num: int, 
                        history: List[Tuple[str, str]]) -> str:
        """Get a move from a model with proper input formatting and validation"""
        # Prepare the input
        model_input = self.prepare_model_input(model, opponent, round_num, history)
        
        # Call the external prediction function
        try:
            prediction = get_prediction(model_input)
            
            # Validate the response
            if prediction.lower() not in ['c', 'd']:
                print(f"Warning: Model {model} returned invalid move '{prediction}'. Defaulting to 'd'.")
                return 'd'
            
            return prediction.lower()
        
        except Exception as e:
            print(f"Error getting prediction from model {model}: {e}. Defaulting to 'd'.")
            return 'd'
    
    def prepare_model_input(self, model: Any, opponent: Any, round_num: int, 
                           history: List[Tuple[str, str]], 
                           all_history: Optional[Dict[str, List[Tuple[str, str]]]] = None) -> Dict[str, Any]:
        """Prepare the input for get_prediction()"""
        # Base task description
        task_description = """You are playing the Prisoner's Dilemma game.

Game Rules:
- You and your opponent simultaneously choose to either Cooperate (C) or Defect (D)
- Scoring:
  - Both Cooperate (C,C): Each player gets 3 points
  - You Cooperate, Opponent Defects (C,D): You get 0 points, Opponent gets 5 points
  - You Defect, Opponent Cooperates (D,C): You get 5 points, Opponent gets 0 points
  - Both Defect (D,D): Each player gets 1 point

Your goal is to maximize your total score across all games.

You must respond with exactly one character: either 'c' (cooperate) or 'd' (defect)."""
        
        # Format game history
        history_str = ""
        if history:
            history_str = "\n\nGame History (your moves, opponent's moves):\n"
            for i, (my_move, opp_move) in enumerate(history, 1):
                history_str += f"Round {i}: You played {my_move.upper()}, Opponent played {opp_move.upper()}\n"
        
        # Include cross-opponent memory if enabled
        cross_memory_str = ""
        if self.enable_cross_memory and all_history is None:
            all_history = self.all_history.get(str(model), {})
        
        if all_history:
            cross_memory_str = "\n\nHistory with other opponents:\n"
            for other_opponent, other_history in all_history.items():
                if str(other_opponent) != str(opponent) and other_history:
                    cross_memory_str += f"\nAgainst {other_opponent}:\n"
                    for i, (my_move, opp_move) in enumerate(other_history[-5:], 1):  # Last 5 games
                        cross_memory_str += f"  You: {my_move.upper()}, Them: {opp_move.upper()}\n"
        
        # Combine all input components
        model_input = {
            "model": model,
            "task": task_description,
            "round": round_num,
            "opponent": str(opponent),
            "history": history_str,
            "cross_memory": cross_memory_str if self.enable_cross_memory else "",
            "prompt": f"{task_description}{history_str}{cross_memory_str}\n\nCurrent round: {round_num}\nOpponent: {opponent}\n\nYour move (c or d):"
        }
        
        return model_input



# Example usage demonstrating how the system works
if __name__ == "__main__":
    # Example models (these would be actual model objects/identifiers)
    example_models = ["gpt-4", "claude-3", "llama-2", "custom-model"]
    
    # Create tournament
    tournament = PrisonersDilemmaTournament(
        models=example_models,
        games_per_matchup=50,
        enable_cross_memory=False
    )
    
    # The following would work once get_prediction is implemented:
    # results = tournament.run_tournament()
    # 
    # print("Tournament Leaderboard:")
    # for rank, (player, score) in enumerate(results.get_leaderboard(), 1):
    #     print(f"{rank}. {player}: {score} points")
    # 
    # print("\nHead-to-Head Matrix:")
    # matrix = results.get_head_to_head_matrix()
    # for player1 in sorted(matrix.keys()):
    #     print(f"\n{player1} vs:")
    #     for player2, stats in matrix[player1].items():
    #         if player1 != player2:
    #             print(f"  {player2}: W-{stats['wins']} L-{stats['losses']} T-{stats['ties']}")
    # 
    # print("\nSummary Statistics:")
    # stats = results.get_summary_stats()
    # print(f"Total games played: {stats['total_games']}")
    # print("\nAverage score per game:")
    # for player, avg in stats['average_score_per_game'].items():
    #     print(f"  {player}: {avg:.2f}")
    # print("\nCooperation rate:")
    # for player, rate in stats['cooperation_rate'].items():
    #     print(f"  {player}: {rate:.2%}")