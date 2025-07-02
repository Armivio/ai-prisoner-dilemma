#!/usr/bin/env python3
"""
Setup script for Prisoner's Dilemma Tournament.
Helps users configure their environment and test API connectivity.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Ensure Python 3.7+ is being used."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required.")
        print(f"   You have: Python {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")


def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("   Try running: pip install -r requirements.txt")
        return False
    return True


def setup_env_file():
    """Create .env file from template if it doesn't exist."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("\n✓ .env file already exists")
        return True
    
    if not env_example_path.exists():
        print("\n❌ .env.example file not found")
        return False
    
    print("\n📝 Creating .env file from template...")
    env_path.write_text(env_example_path.read_text())
    print("✓ .env file created")
    print("⚠️  Please edit .env and add your API keys!")
    return True


def test_api_setup():
    """Test if APIs are properly configured."""
    print("\n🔍 Testing API configuration...")
    
    try:
        from api_models import test_api_connectivity
        test_api_connectivity()
    except ImportError:
        print("❌ Could not import api_models. Make sure dependencies are installed.")
        return False
    
    return True


def run_minimal_test():
    """Run a minimal test if APIs are configured."""
    print("\n🎮 Would you like to run a minimal test tournament? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("\nRunning minimal test...")
        subprocess.run([sys.executable, "example_api_tournament.py", "--test"])


def main():
    """Main setup process."""
    print("🎯 Prisoner's Dilemma Tournament Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies():
        print("\n⚠️  Setup incomplete. Please install dependencies manually.")
        return
    
    # Setup environment file
    if not setup_env_file():
        print("\n⚠️  Setup incomplete. Please create .env file manually.")
        return
    
    # Test API setup
    test_api_setup()
    
    # Offer to run test
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env and add your API keys")
    print("2. Run: python example_api_tournament.py --test")
    print("3. Run: python example_api_tournament.py")
    
    # Check if any API keys are already set
    from dotenv import load_dotenv
    load_dotenv()
    
    has_keys = any([
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GOOGLE_API_KEY")
    ])
    
    if has_keys:
        run_minimal_test()


if __name__ == "__main__":
    main()