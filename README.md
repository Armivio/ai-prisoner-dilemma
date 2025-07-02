# Prisoner's Dilemma Tournament System

A Python implementation of a Prisoner's Dilemma tournament system designed to work with any AI model that follows a simple interface.

## Features

- **Round-robin tournament format**: Each AI model plays against every other model
- **Real AI model support**: Supports models from Anthropic, OpenAI, and Google
- **Configurable game parameters**: Set number of games per matchup
- **Cross-opponent memory**: Optional feature to let AIs remember games against other opponents
- **Comprehensive results tracking**: Scores, head-to-head results, and detailed statistics
- **Clean model interface**: Easy integration with any AI model using exact API names
- **Robust error handling**: Gracefully handles API errors, quota limits, and invalid responses

## Game Rules

The Prisoner's Dilemma scoring matrix:
- **Both Cooperate (C,C)**: Each player gets 3 points
- **Cooperate vs Defect (C,D)**: Cooperator gets 0 points, Defector gets 5 points  
- **Both Defect (D,D)**: Each player gets 1 point

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/prisoners-dilemma-tournament.git
cd prisoners-dilemma-tournament

# Run the setup script
python setup.py

# Edit .env and add your API keys
# Then run a test tournament
python example_api_tournament.py --test
```

## Manual Installation

```bash
# Install dependencies for API support
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Basic Tournament

```python
from prisoners_dilemma_tournament import PrisonersDilemmaTournament

# Define your models
models = ["model1", "model2", "model3", "model4"]

# Create and run tournament
tournament = PrisonersDilemmaTournament(
    models=models,
    games_per_matchup=100,
    enable_cross_memory=False
)

results = tournament.run_tournament()

# View results
print(results.get_leaderboard())
print(results.get_head_to_head_matrix())
```

### Using Real AI Models

The tournament system comes with built-in support for popular AI models using their exact API names:

```python
# Supported models include:
# - Anthropic: claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022, claude-sonnet-4-20250514, claude-opus-4-20250514
# - OpenAI: gpt-4o-mini
# - Google: NOT YET TESTED

# Example with real models
from prisoners_dilemma_tournament import PrisonersDilemmaTournament

models = ["claude-3-5-haiku-20241022", "gpt-4o-mini"]
tournament = PrisonersDilemmaTournament(
    models=models,
    games_per_matchup=20,
    enable_cross_memory=True
)

results = tournament.run_tournament()
```

### Running an API Tournament

```bash
# Quick test with 2 games
python example_api_tournament.py --test

# Full tournament (includes Claude 3.5 Haiku, Claude 3.7 Sonnet, Claude Sonnet 4, Claude Opus 4, GPT-4o Mini)
python example_api_tournament.py
```

### Custom Model Integration

To add your own model or API, implement the `get_prediction` function in api_models.py:

```python
def get_prediction(model_input):
    """
    Interface with your AI model.
    
    Args:
        model_input: Dict containing:
            - model: The model identifier
            - prompt: Complete formatted prompt for the model
            - task: Task description
            - round: Current round number
            - opponent: Opponent identifier
            - history: Game history string
            - cross_memory: Cross-opponent memory (if enabled)
    
    Returns:
        str: Either "c" (cooperate) or "d" (defect)
    """
    # Your implementation here
    # Example: return call_ai_api(model_input["model"], model_input["prompt"])
    pass
```

### Model Input Format

Each AI model receives:
1. **Task description**: Clear explanation of the game and scoring
2. **Current round**: Round number (without revealing total rounds)
3. **Game history**: Previous moves in current matchup
4. **Opponent identifier**: Full model name of current opponent
5. **Cross-opponent memory** (optional): History from other matchups

**Note**: Models are identified by their exact API names (e.g., `claude-sonnet-4-20250514`, `gpt-4o-mini`) rather than simplified names.

### Results Analysis

The tournament provides comprehensive results:

## Example Strategies

See `example_usage.py` for demonstrations of various strategies:
- Always Cooperate
- Always Defect
- Tit-for-Tat
- Random

## Architecture

### Core Classes

1. **PrisonersDilemmaTournament**: Main tournament manager
   - Schedules matchups
   - Manages game execution
   - Handles model communication

2. **TournamentResults**: Results tracking and reporting
   - Tracks scores
   - Stores game history
   - Generates statistics

3. **GameResult**: Individual game outcome
   - Players and moves
   - Scores
   - Round number

4. **MatchupHistory**: History between two players
   - All games played
   - Move sequences

## Advanced Features

### Cross-Opponent Memory

Enable models to remember games against other opponents:

```python
tournament = PrisonersDilemmaTournament(
    models=models,
    games_per_matchup=100,
    enable_cross_memory=True  # Enable memory across opponents
)
```

### Custom Scoring Matrix

Modify the scoring matrix if needed:

```python
tournament.SCORING_MATRIX = {
    ('c', 'c'): (3, 3),
    ('c', 'd'): (0, 5),
    ('d', 'c'): (5, 0),
    ('d', 'd'): (1, 1),
}
```

## Error Handling

The system should handle various error cases:
- Invalid model responses (defaults to defect)
- Model timeouts or failures
- Malformed responses


### Estimated Costs

For a tournament with 6 models and 20 games per matchup:
- Total games: 6 × 5 ÷ 2 × 20 = 300 games
- API calls: 600 (2 per game)
- Approximate cost: $0.10 - $1.00 depending on models used

## Troubleshooting

### Common Issues

1. **ImportError for API packages**
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Not Found**
   - Make sure you've created `.env` from `.env.example`
   - Check that API keys are correctly set in `.env`

3. **Model Not Responding Correctly**
   - The system defaults to "defect" for invalid responses
   - Check API rate limits if seeing many defects

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License