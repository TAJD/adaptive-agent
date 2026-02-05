#!/usr/bin/env python3
"""
Demo Seeding Script

Pre-seeds the episodic memory with a known failure episode so demos
are reliable and don't depend on LLM randomness.

This creates a realistic learning scenario:
- A prior session encountered an FX Impact query for Product C
- The agent failed initially (wrong filtering)
- The agent then succeeded with the correct approach
- This episode is now stored for future sessions to learn from

Usage:
    uv run python scripts/seed_demo.py          # Seed with default episode
    uv run python scripts/seed_demo.py --clear  # Clear and reseed
    uv run python scripts/seed_demo.py --show   # Show current episodes
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.file import FileStorage
from src.strategies.episodic_memory import Episode, extract_keywords


def get_storage() -> FileStorage:
    """Get the unified storage instance (same as chat.py --persist-default)."""
    storage_path = Path(__file__).parent.parent / ".agent_memory"
    storage_path.mkdir(parents=True, exist_ok=True)
    return FileStorage(storage_path)


def create_fx_episode() -> Episode:
    """
    Create a realistic FX Impact episode that teaches:
    - How to filter by FSLine Statement L2 = 'Foreign Exchange Gain/Loss'
    - How to aggregate across all countries
    - How to handle the result (can be positive or negative)
    """
    query = "What was the total Foreign Exchange Gain/Loss (FX impact) for Product C across ALL countries in fiscal year 2024? This is found in FSLine Statement L2. Return just the numeric value (can be negative)."

    failed_code = """# Initial attempt - incorrect filtering
result = data[
    (data['Product'] == 'Product C') &
    (data['Fiscal Year'] == 2024)
]['Amount in USD'].sum()
"""

    fixed_code = """# Correct approach: filter by FSLine Statement L2 for FX impact
fx_data = data[
    (data['Product'] == 'Product C') &
    (data['Fiscal Year'] == 2024) &
    (data['FSLine Statement L2'] == 'Foreign Exchange Gain/Loss')
]
result = fx_data['Amount in USD'].sum()
"""

    keywords = extract_keywords(query)

    return Episode(
        query=query,
        failed_code=failed_code,
        error_message="Wrong value. Expected 35095.60, got much larger value. Missing FSLine Statement L2 filter.",
        fixed_code=fixed_code,
        error_type="large_numeric_error",
        keywords=keywords,
        task_id="fx-impact-product-c-2024",
        timestamp="1700000000000",  # Fixed timestamp for consistent key
        version="1.0",
        change_description="Added FSLine Statement L2 filter for accurate FX impact calculation",
        effectiveness_score=0.7,  # High effectiveness - this fix works
        times_applied=3,
        times_succeeded=2,
    )


def create_revenue_episode() -> Episode:
    """
    Create a revenue calculation episode that teaches:
    - How to filter by multiple dimensions (Product, Quarter, Year)
    - How to use FSLine Statement L2 for specific line items
    """
    query = "What was the total Gross Revenue for Product A in Q1 2023?"

    failed_code = """# Initial attempt - missing specific line item filter
result = data[
    (data['Product'] == 'Product A') &
    (data['Fiscal Quarter'] == 'Q1') &
    (data['Fiscal Year'] == 2023)
]['Amount in USD'].sum()
"""

    fixed_code = """# Correct: filter by FSLine Statement L2 = 'Gross Revenue'
revenue_data = data[
    (data['Product'] == 'Product A') &
    (data['Fiscal Quarter'] == 'Q1') &
    (data['Fiscal Year'] == 2023) &
    (data['FSLine Statement L2'] == 'Gross Revenue')
]
result = revenue_data['Amount in USD'].sum()
"""

    keywords = extract_keywords(query)

    return Episode(
        query=query,
        failed_code=failed_code,
        error_message="Wrong value. Missing FSLine Statement L2 filter - summed all line items instead of just Gross Revenue.",
        fixed_code=fixed_code,
        error_type="much_larger",
        keywords=keywords,
        task_id="revenue-product-a-q1-2023",
        timestamp="1700000001000",
        version="1.0",
        change_description="Added FSLine Statement L2 filter for specific revenue calculation",
        effectiveness_score=0.8,
        times_applied=5,
        times_succeeded=4,
    )


def seed_episodes(storage: FileStorage, clear_first: bool = False) -> None:
    """Seed the storage with learning episodes."""
    if clear_first:
        print("Clearing existing episodes...")
        for key in storage.list_keys("episodes"):
            storage.delete(key)
        print("Cleared.")

    episodes = [
        create_fx_episode(),
        create_revenue_episode(),
    ]

    print(f"\nSeeding {len(episodes)} episodes...")

    for episode in episodes:
        # Generate key matching EpisodicMemoryStrategy format
        import hashlib
        query_hash = hashlib.md5(episode.query.encode()).hexdigest()[:8]
        key = f"episodes/{episode.task_id}/{query_hash}/{episode.timestamp}"

        storage.save(key, episode.to_dict())
        print(f"  Saved: {episode.task_id}")
        print(f"    Keywords: {episode.keywords[:5]}...")
        print(f"    Has fix: {episode.fixed_code is not None}")
        print(f"    Effectiveness: {episode.effectiveness_score:.0%}")

    print(f"\nSeeding complete! Total episodes: {len(storage.list_keys('episodes'))}")


def show_episodes(storage: FileStorage) -> None:
    """Display all stored episodes."""
    keys = storage.list_keys("episodes")

    if not keys:
        print("No episodes stored.")
        return

    print(f"\n{'='*70}")
    print(f"  STORED EPISODES ({len(keys)} total)")
    print(f"{'='*70}\n")

    for key in keys:
        data = storage.load(key)
        if not data:
            continue

        episode = Episode.from_dict(data)
        print(f"Key: {key}")
        print(f"  Query: {episode.query[:60]}...")
        print(f"  Keywords: {episode.keywords[:5]}")
        print(f"  Has Fix: {episode.fixed_code is not None}")
        print(f"  Error Type: {episode.error_type}")
        print(f"  Effectiveness: {episode.effectiveness_score:.0%}")
        print(f"  Applied/Succeeded: {episode.times_applied}/{episode.times_succeeded}")
        print()

        if episode.fixed_code:
            print("  Fixed Code Preview:")
            for line in episode.fixed_code.split('\n')[:5]:
                print(f"    {line}")
            print()


def show_episode_json(storage: FileStorage) -> None:
    """Display raw JSON of first episode for demo purposes."""
    keys = storage.list_keys("episodes")
    if not keys:
        print("No episodes to display.")
        return

    data = storage.load(keys[0])
    print("\n" + "="*70)
    print("  EPISODE JSON (for demo evidence)")
    print("="*70 + "\n")
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Seed demo with learning episodes")
    parser.add_argument("--clear", action="store_true", help="Clear existing episodes first")
    parser.add_argument("--show", action="store_true", help="Show current episodes")
    parser.add_argument("--json", action="store_true", help="Show raw JSON of first episode")
    args = parser.parse_args()

    storage = get_storage()
    storage_path = Path(__file__).parent.parent / ".agent_memory"

    print(f"Storage path: {storage_path}")

    if args.show:
        show_episodes(storage)
        return

    if args.json:
        show_episode_json(storage)
        return

    seed_episodes(storage, clear_first=args.clear)

    print("\n" + "-"*70)
    print("Next steps:")
    print("  1. Run: uv run python scripts/demo_cross_session.py")
    print("  2. Session 2 will retrieve the seeded episode")
    print("  3. Show evidence: uv run python scripts/seed_demo.py --json")
    print("-"*70 + "\n")


if __name__ == "__main__":
    main()
