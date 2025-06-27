"""
Fantasy-football missing data filler for K and DST positions
-----------------------------------------------------------

Fills in -1 values in training_data_{year}.csv files:
  * fantasy_points: Uses TTL column from stats files
  * games_played: For kickers only, counts non-BYE/non-empty weeks 1-17
  * Uses fuzzy name matching consistent with original preprocessing
"""

import pandas as pd
from pathlib import Path
import logging
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class NameMatcher:
    """Utility for consistent player name handling (matches original script)."""

    @staticmethod
    def normalize_name(name):
        """Lowercase, remove suffixes/punctuation/extra spaces."""
        if pd.isna(name):
            return ""
        name = str(name).lower()
        name = re.sub(r"\s+(jr\.?|sr\.?|iii|ii|iv|v)$", "", name, flags=re.IGNORECASE)
        name = (
            name.replace(".", "")
            .replace("'", "")
            .replace("`", "")
            .replace("-", " ")
        )
        return " ".join(name.split())

    @staticmethod
    def create_match_key(name):
        """Create a key from the normalized name only (no position)."""
        normalized = NameMatcher.normalize_name(name)
        return re.sub(r"[^a-z0-9]", "", normalized)

    @staticmethod
    def similarity_score(name1, name2):
        n1, n2 = map(NameMatcher.normalize_name, (name1, name2))
        return SequenceMatcher(None, n1, n2).ratio()


def count_kicker_games_played(stats_row):
    """
    Count games played for kickers by examining weeks 1-17.
    Excludes BYE weeks and empty/missing values.
    """
    week_cols = [str(i) for i in range(1, 18)]  # Weeks 1-17
    games_count = 0
    
    for week in week_cols:
        if week in stats_row.index:
            value = str(stats_row[week]).strip()
            if value and value != '' and value.upper() != 'BYE' and value != '-':
                try:
                    # Try to convert to float to ensure it's a valid score
                    float(value)
                    games_count += 1
                except ValueError:
                    # Skip non-numeric values that aren't BYE
                    continue
    
    return games_count


def find_best_match(target_name, stats_df, threshold=0.8):
    """
    Find best matching player using fuzzy string matching.
    Returns the matching row from stats_df or None.
    """
    target_normalized = NameMatcher.normalize_name(target_name)
    best_score = 0.0
    best_match = None
    
    for _, stats_row in stats_df.iterrows():
        score = NameMatcher.similarity_score(target_name, stats_row['Player'])
        if score > best_score and score >= threshold:
            best_score = score
            best_match = stats_row
    
    return best_match


def fill_missing_data_for_year(year):
    """Process missing data for a specific year."""
    logging.info(f"[{year}] Processing missing data...")
    
    # File paths
    training_file = Path(f"data/processed/training_data_{year}.csv")
    k_stats_file = Path(f"data/raw/stats/stats_{year}_k.csv")
    dst_stats_file = Path(f"data/raw/stats/stats_{year}_dst.csv")
    
    # Check if training file exists
    if not training_file.exists():
        logging.warning(f"[{year}] Training file not found: {training_file}")
        return
    
    # Load training data
    training_df = pd.read_csv(training_file)
    logging.info(f"[{year}] Loaded training data: {len(training_df)} rows")
    
    # Track updates
    fantasy_points_updated = 0
    games_played_updated = 0
    
    # Process Kickers
    if k_stats_file.exists():
        logging.info(f"[{year}] Processing kickers from {k_stats_file}")
        k_stats_df = pd.read_csv(k_stats_file)
        
        # Get kickers with missing data
        kicker_mask = (training_df['position'] == 'K') & (
            (training_df['fantasy_points'] == -1) | 
            (training_df['games_played'] == -1) |
            training_df['fantasy_points'].isna() |
            training_df['games_played'].isna()
        )
        kickers_to_update = training_df[kicker_mask]
        
        logging.info(f"[{year}] Found {len(kickers_to_update)} kickers needing updates")
        
        for idx, kicker in kickers_to_update.iterrows():
            best_match = find_best_match(kicker['name'], k_stats_df)
            
            if best_match is not None:
                # Update fantasy points if missing
                if pd.isna(kicker['fantasy_points']) or kicker['fantasy_points'] == -1:
                    try:
                        ttl_value = float(best_match['TTL'])
                        training_df.at[idx, 'fantasy_points'] = ttl_value
                        fantasy_points_updated += 1
                        logging.info(f"  {kicker['name']} -> {best_match['Player']}: {ttl_value} fantasy points")
                    except (ValueError, TypeError) as e:
                        logging.warning(f"  Could not convert TTL for {best_match['Player']}: {e}")
                
                # Update games played if missing
                if pd.isna(kicker['games_played']) or kicker['games_played'] == -1:
                    games_count = count_kicker_games_played(best_match)
                    training_df.at[idx, 'games_played'] = games_count
                    games_played_updated += 1
                    logging.info(f"  {kicker['name']} games played: {games_count}")
            else:
                logging.warning(f"  No match found for kicker: {kicker['name']}")
    else:
        logging.warning(f"[{year}] Kicker stats file not found: {k_stats_file}")
    
    # Process DST
    if dst_stats_file.exists():
        logging.info(f"[{year}] Processing DST from {dst_stats_file}")
        dst_stats_df = pd.read_csv(dst_stats_file)
        
        # Get DST with missing fantasy points
        dst_mask = (training_df['position'] == 'DST') & (
            (training_df['fantasy_points'] == -1) | 
            training_df['fantasy_points'].isna()
        )
        dsts_to_update = training_df[dst_mask]
        
        logging.info(f"[{year}] Found {len(dsts_to_update)} DST needing updates")
        
        for idx, dst in dsts_to_update.iterrows():
            best_match = find_best_match(dst['name'], dst_stats_df)
            
            if best_match is not None:
                # Update fantasy points
                try:
                    ttl_value = float(best_match['TTL'])
                    training_df.at[idx, 'fantasy_points'] = ttl_value
                    fantasy_points_updated += 1
                    logging.info(f"  {dst['name']} -> {best_match['Player']}: {ttl_value} fantasy points")
                except (ValueError, TypeError) as e:
                    logging.warning(f"  Could not convert TTL for {best_match['Player']}: {e}")
            else:
                logging.warning(f"  No match found for DST: {dst['name']}")
    else:
        logging.warning(f"[{year}] DST stats file not found: {dst_stats_file}")
    
    # Create backup and save updated data
    backup_file = training_file.with_suffix('.csv.backup')
    if not backup_file.exists():
        training_df_original = pd.read_csv(training_file)
        training_df_original.to_csv(backup_file, index=False)
        logging.info(f"[{year}] Created backup: {backup_file}")
    
    training_df.to_csv(training_file, index=False)
    logging.info(f"[{year}] Updated training data saved")
    logging.info(f"[{year}] Fantasy points updated: {fantasy_points_updated}")
    logging.info(f"[{year}] Games played updated: {games_played_updated}")


def fill_missing_data_all_years():
    """Fill missing data for all available years (2021-2024)."""
    logging.info("Starting missing data fill process...")
    
    years = [2021, 2022, 2023, 2024]
    
    for year in years:
        try:
            fill_missing_data_for_year(year)
        except Exception as e:
            logging.error(f"[{year}] Error processing: {e}")
    
    logging.info("Missing data fill process complete!")


if __name__ == "__main__":
    fill_missing_data_all_years()