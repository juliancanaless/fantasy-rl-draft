"""
Fantasy-football ADP + stats preprocessing
-----------------------------------------

Changes from the original:
  * Strips digits from position codes (RB1 -> RB)
  * match_key is built from the player name only (no position suffix)
  * After fuzzy matching, any player still without stats gets –1 in both
    fantasy_points and games_played
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
    """Convert 'RB1' -> 'RB'. Keeps DST, K, etc. Returns upper-case or None."""
    if pd.isna(pos):
        return None
    return re.sub(r"\d+", "", str(pos)).upper()


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
# ADP cleaning
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
    logging.info(f"[{year}] ADP cleaned rows: {len(df)}")
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
# Pipeline driver
# ------------------------------------------------------------------

def process_all_years():
    adp_dir = Path("data/raw/adp")
    stats_dir = Path("data/raw/stats")
    out_dir = Path("data/processed")

    # 1. Clean ADP
    for yr in range(2021, 2026):
        src = adp_dir / f"adp_{yr}.csv"
        if src.exists():
            clean_adp_data(src, out_dir / f"adp_{yr}_clean.csv", yr)
        else:
            logging.warning(f"[{yr}] ADP file missing: {src}")

    # 2. Merge ADP with stats (only for years that have stats)
    for yr in range(2021, 2025):
        adp_path = out_dir / f"adp_{yr}_clean.csv"
        stats_path = stats_dir / f"stats_{yr}.csv"
        if not adp_path.exists() or not stats_path.exists():
            logging.warning(f"[{yr}] Skipping merge (missing file)")
            continue

        adp_df = pd.read_csv(adp_path)
        stats_df = pd.read_csv(stats_path)
        merged = merge_adp_stats(adp_df, stats_df, yr)
        merged.to_csv(out_dir / f"training_data_{yr}.csv", index=False)
        logging.info(f"[{yr}] Saved training_data to processed directory")


if __name__ == "__main__":
    process_all_years()
