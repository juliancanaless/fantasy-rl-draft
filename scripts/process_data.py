"""
Fantasy-football Complete Data Processing Pipeline
-------------------------------------------------

Combines preprocessing + missing data filling with strict position filtering.
Only keeps QB, RB, WR, TE, K, DST positions.
Sets all DST games_played to 16.
"""

import pandas as pd
from pathlib import Path
import logging
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def strip_position_code(pos):
    """Convert 'RB1' -> 'RB', 'DT' -> 'DST'. Returns upper-case or None."""
    if pd.isna(pos):
        return None
    
    # Clean position and handle variations
    pos_clean = re.sub(r"\d+", "", str(pos)).upper().strip()
    
    # Handle defense variations
    if pos_clean in ['DEF', 'D/ST', 'D']:
        return 'DST'
    
    return pos_clean


class NameMatcher:
    """Utility for consistent player name handling."""

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


# ------------------------------------------------------------------
# ADP cleaning with strict position filtering
# ------------------------------------------------------------------

def clean_adp_data(input_path, output_path, year):
    """Standardize raw ADP CSV and write cleaned file."""
    logging.info(f"[{year}] Cleaning ADP -> {output_path}")
    df = pd.read_csv(input_path).dropna(subset=["Player"])

    df["Team"] = df["Team"].fillna("FA")
    df["Bye"] = (
        df["Bye"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .fillna(0)
    )

    df["POS"] = df["POS"].fillna("UNK").apply(strip_position_code)

    # STRICT POSITION FILTERING
    VALID_POSITIONS = {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}
    original_count = len(df)
    df = df[df["POS"].isin(VALID_POSITIONS)]
    filtered_count = original_count - len(df)
    
    if filtered_count > 0:
        logging.info(f"[{year}] Filtered out {filtered_count} players with invalid positions")

    keep_cols = [c for c in ("Player", "Team", "Bye", "POS", "AVG") if c in df.columns]
    df = df[keep_cols]
    if "AVG" in df.columns:
        df["AVG"] = df["AVG"].fillna(1000)

    df = df.rename(
        columns={
            "Player": "name",
            "Team": "team",
            "Bye": "bye_week",
            "POS": "position",
            "AVG": "adp",
        }
    )

    df["name_normalized"] = df["name"].apply(NameMatcher.normalize_name)
    df["match_key"] = df["name"].apply(NameMatcher.create_match_key)
    df["year"] = year
    df = df.sort_values("adp").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"[{year}] ADP cleaned rows: {len(df)}, positions: {sorted(df['position'].unique())}")
    return df


# ------------------------------------------------------------------
# Merge ADP with season stats
# ------------------------------------------------------------------

def merge_adp_stats(adp_df, stats_df, year):
    """Return ADP rows with fantasy_points and games_played columns merged in."""
    logging.info(f"[{year}] Merging ADP with stats")

    stats_df = stats_df.rename(
        columns={
            "Name": "name",
            "POS": "position",
            "PTS": "fantasy_points",
            "GP": "games_played",
        }
    )
    stats_df["position"] = stats_df["position"].apply(strip_position_code)

    # Filter stats to valid positions only
    VALID_POSITIONS = {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}
    stats_df = stats_df[stats_df["position"].isin(VALID_POSITIONS)]

    adp_df["position"] = adp_df["position"].apply(strip_position_code)
    adp_df["match_key"] = adp_df["name"].apply(NameMatcher.create_match_key)
    stats_df["match_key"] = stats_df["name"].apply(NameMatcher.create_match_key)

    merged = pd.merge(
        adp_df,
        stats_df[["match_key", "position", "fantasy_points", "games_played"]],
        on="match_key",
        how="left",
        suffixes=("", "_stats"),
    )

    pos_mask = merged["position_stats"].notna()
    merged.loc[pos_mask, "position"] = merged.loc[pos_mask, "position_stats"]
    merged = merged.drop(columns=["position_stats"])

    # Fuzzy match for remaining high-ADP players
    still_na = merged["fantasy_points"].isna() & (merged["adp"] < 200)
    if still_na.any():
        candidates = merged[still_na]
        logging.info(f"[{year}] Fuzzy matching {len(candidates)} high-value players")
        for idx, row in candidates.iterrows():
            same_pos = stats_df[stats_df["position"] == row["position"]]
            best_score, best_row = 0.0, None
            for _, cand in same_pos.iterrows():
                score = NameMatcher.similarity_score(row["name"], cand["name"])
                if score > best_score and score >= 0.8:
                    best_score, best_row = score, cand
            if best_row is not None:
                merged.at[idx, "fantasy_points"] = best_row["fantasy_points"]
                merged.at[idx, "games_played"] = best_row["games_played"]

    merged["fantasy_points"] = pd.to_numeric(merged["fantasy_points"], errors="coerce")
    merged["games_played"] = pd.to_numeric(merged["games_played"], errors="coerce")

    final_unmatched = merged["fantasy_points"].isna().sum()
    if final_unmatched:
        logging.warning(
            f"[{year}] {final_unmatched} players unmatched. "
            "Setting fantasy_points and games_played to -1."
        )
        merged["fantasy_points"] = merged["fantasy_points"].fillna(-1.0)
        merged["games_played"] = merged["games_played"].fillna(-1)

    return merged


# ------------------------------------------------------------------
# Missing data filling functions
# ------------------------------------------------------------------

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


def fill_missing_data_for_year(merged_df, year):
    """Fill missing data for K and DST positions."""
    logging.info(f"[{year}] Filling missing data...")
    
    # File paths for detailed stats
    k_stats_file = Path(f"data/raw/stats/stats_{year}_k.csv")
    dst_stats_file = Path(f"data/raw/stats/stats_{year}_dst.csv")
    
    # Track updates
    fantasy_points_updated = 0
    games_played_updated = 0
    
    # Process Kickers
    if k_stats_file.exists():
        logging.info(f"[{year}] Processing kickers from {k_stats_file}")
        k_stats_df = pd.read_csv(k_stats_file)
        
        # Get kickers with missing data
        kicker_mask = (merged_df['position'] == 'K') & (
            (merged_df['fantasy_points'] == -1) | 
            (merged_df['games_played'] == -1) |
            merged_df['fantasy_points'].isna() |
            merged_df['games_played'].isna()
        )
        kickers_to_update = merged_df[kicker_mask]
        
        logging.info(f"[{year}] Found {len(kickers_to_update)} kickers needing updates")
        
        for idx, kicker in kickers_to_update.iterrows():
            best_match = find_best_match(kicker['name'], k_stats_df)
            
            if best_match is not None:
                # Update fantasy points if missing
                if pd.isna(kicker['fantasy_points']) or kicker['fantasy_points'] == -1:
                    try:
                        ttl_value = float(best_match['TTL'])
                        merged_df.at[idx, 'fantasy_points'] = ttl_value
                        fantasy_points_updated += 1
                    except (ValueError, TypeError):
                        pass
                
                # Update games played if missing
                if pd.isna(kicker['games_played']) or kicker['games_played'] == -1:
                    games_count = count_kicker_games_played(best_match)
                    merged_df.at[idx, 'games_played'] = games_count
                    games_played_updated += 1
    
    # Process DST
    if dst_stats_file.exists():
        logging.info(f"[{year}] Processing DST from {dst_stats_file}")
        dst_stats_df = pd.read_csv(dst_stats_file)
        
        # Get DST with missing fantasy points
        dst_mask = (merged_df['position'] == 'DST') & (
            (merged_df['fantasy_points'] == -1) | 
            merged_df['fantasy_points'].isna()
        )
        dsts_to_update = merged_df[dst_mask]
        
        logging.info(f"[{year}] Found {len(dsts_to_update)} DST needing updates")
        
        for idx, dst in dsts_to_update.iterrows():
            best_match = find_best_match(dst['name'], dst_stats_df)
            
            if best_match is not None:
                # Update fantasy points
                try:
                    ttl_value = float(best_match['TTL'])
                    merged_df.at[idx, 'fantasy_points'] = ttl_value
                    fantasy_points_updated += 1
                except (ValueError, TypeError):
                    pass
    
    # SET ALL DST GAMES PLAYED TO 16
    dst_mask = merged_df['position'] == 'DST'
    dst_count = dst_mask.sum()
    merged_df.loc[dst_mask, 'games_played'] = 16
    logging.info(f"[{year}] Set {dst_count} DST players to 16 games played")
    
    logging.info(f"[{year}] Fantasy points updated: {fantasy_points_updated}")
    logging.info(f"[{year}] Games played updated: {games_played_updated}")
    
    return merged_df


# ------------------------------------------------------------------
# Complete pipeline
# ------------------------------------------------------------------

def process_all_years():
    """Complete data processing pipeline."""
    adp_dir = Path("data/raw/adp")
    stats_dir = Path("data/raw/stats")
    out_dir = Path("data/processed")
    
    VALID_POSITIONS = {'QB', 'RB', 'WR', 'TE', 'K', 'DST'}

    # 1. Clean ADP files
    logging.info("=== PHASE 1: CLEANING ADP DATA ===")
    for yr in range(2021, 2026):
        src = adp_dir / f"adp_{yr}.csv"
        if src.exists():
            clean_adp_data(src, out_dir / f"adp_{yr}_clean.csv", yr)
        else:
            logging.warning(f"[{yr}] ADP file missing: {src}")

    # 2. Merge ADP with stats AND fill missing data
    logging.info("=== PHASE 2: MERGING & FILLING DATA ===")
    for yr in range(2021, 2025):
        adp_path = out_dir / f"adp_{yr}_clean.csv"
        stats_path = stats_dir / f"stats_{yr}.csv"
        
        if not adp_path.exists() or not stats_path.exists():
            logging.warning(f"[{yr}] Skipping merge (missing file)")
            continue

        adp_df = pd.read_csv(adp_path)
        stats_df = pd.read_csv(stats_path)
        
        # Merge stats
        merged = merge_adp_stats(adp_df, stats_df, yr)
        
        # Fill missing data
        merged = fill_missing_data_for_year(merged, yr)
        
        # Final position filter (safety check)
        original_count = len(merged)
        merged = merged[merged['position'].isin(VALID_POSITIONS)]
        final_count = len(merged)
        
        if original_count != final_count:
            logging.warning(f"[{yr}] Final filter removed {original_count - final_count} invalid positions")
        
        # Save final training data
        output_path = out_dir / f"training_data_{yr}.csv"
        merged.to_csv(output_path, index=False)
        
        # Summary
        pos_counts = merged['position'].value_counts()
        logging.info(f"[{yr}] FINAL DATA - Total: {len(merged)} players")
        for pos in sorted(VALID_POSITIONS):
            count = pos_counts.get(pos, 0)
            logging.info(f"  {pos}: {count}")
        
        logging.info(f"[{yr}] Saved to: {output_path}")

    logging.info("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    process_all_years()