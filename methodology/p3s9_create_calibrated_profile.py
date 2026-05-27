"""
p3s9_create_calibrated_profile.py

Phase 3, step 9.

Purpose:
    Convert regression coefficients into a calibrated OSRM Lua profile.

Inputs:
    runs/RunXXX/results/regression_coefficients.csv
    runs/RunXXX/ambulance_nl.lua

Outputs:
    runs/RunXXX/ambulance_nl_calibrated.lua
    runs/RunXXX/calibrated_profile.diff
    runs/RunXXX/results/calibration_changes.csv

Method:
    Road-class coefficients are interpreted as seconds per kilometre and
    translated into updated road speeds. The turn coefficient is interpreted
    as seconds per turn and added to the Lua turn_penalty.
"""

import difflib
import re

import polars as pl

from paths import (
    ACCESS_LUA,
    CALIBRATED_LUA,
    CALIBRATED_DIFF,
    REGRESSION_COEFFICIENTS_CSV,
    CALIBRATION_CHANGES_CSV,
)

from configs import (
    ROAD_COEF_MAP,
    BASE_ROAD_SPEED_KMH,
    MIN_CALIBRATED_SPEED_KMH,
    MAX_CALIBRATED_SPEED_KMH,
    APPLY_ONLY_SIGNIFICANT_COEFS,
    P_VALUE_THRESHOLD,
)


# =========================================================
# Input checks
# =========================================================

def check_inputs() -> None:
    """
    Validate required calibration inputs.
    """
    if not REGRESSION_COEFFICIENTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing regression coefficients: {REGRESSION_COEFFICIENTS_CSV}"
        )

    if not ACCESS_LUA.exists():
        raise FileNotFoundError(
            f"Missing ambulance access Lua profile: {ACCESS_LUA}"
        )


# =========================================================
# Coefficient filtering
# =========================================================

def filter_coefficients(
    coef_df: pl.DataFrame,
    significant_only: bool = False,
    alpha: float = 0.05,
) -> pl.DataFrame:
    """
    Keep only coefficients used for calibration.
    """

    required_cols = ["variable", "coefficient", "p_value"]
    missing = [col for col in required_cols if col not in coef_df.columns]

    if missing:
        raise ValueError(
            f"Regression coefficient table is missing required columns: {missing}"
        )

    valid_variables = list(ROAD_COEF_MAP.keys()) + ["n_turns"]

    coef_df = coef_df.filter(
        pl.col("variable").is_in(valid_variables)
    )

    if significant_only:
        coef_df = coef_df.filter(
            pl.col("p_value") <= alpha
        )

    return coef_df


def build_calibration_lookup(coef_df: pl.DataFrame) -> dict[str, float]:
    """
    Convert coefficient table into lookup dictionary.
    """

    return {
        row["variable"]: float(row["coefficient"])
        for row in coef_df.iter_rows(named=True)
    }


# =========================================================
# Calibration helpers
# =========================================================

def convert_speed(old_speed_kmh: float, beta_sec_per_km: float) -> float:
    """
    Convert baseline road speed using a regression coefficient.

    Positive beta means OSRM is too fast on that road class, so the new speed
    is lower. Negative beta means OSRM is too slow, so the new speed is higher.
    """

    old_sec_per_km = 3600 / old_speed_kmh
    new_sec_per_km = old_sec_per_km + beta_sec_per_km

    if new_sec_per_km <= 0:
        return MAX_CALIBRATED_SPEED_KMH

    new_speed = 3600 / new_sec_per_km

    return max(
        MIN_CALIBRATED_SPEED_KMH,
        min(MAX_CALIBRATED_SPEED_KMH, new_speed),
    )


def replace_lua_speed(lua: str, road_class: str, new_speed: float) -> str:
    """
    Replace one Lua road-speed entry.

    The regex is anchored to line start to avoid accidental replacement of
    unrelated occurrences of the road-class name.
    """

    pattern = rf"(^\s*{re.escape(road_class)}\s*=\s*)[0-9.]+"
    replacement = rf"\g<1>{new_speed:.2f}"

    lua_new, n = re.subn(
        pattern,
        replacement,
        lua,
        count=1,
        flags=re.MULTILINE,
    )

    if n != 1:
        raise RuntimeError(
            f"Expected to replace exactly one speed entry for road class "
            f"{road_class!r}, but replaced {n}."
        )

    return lua_new


def replace_turn_penalty(lua: str, beta_turn: float) -> tuple[str, float]:
    """
    Replace standalone turn_penalty.

    This intentionally does not match u_turn_penalty.
    """

    pattern = r"(^\s*turn_penalty\s*=\s*)([0-9.]+)"

    match = re.search(
        pattern,
        lua,
        flags=re.MULTILINE,
    )

    if not match:
        raise RuntimeError(
            "Could not find standalone turn_penalty in Lua profile."
        )

    old_penalty = float(match.group(2))
    new_penalty = max(0, old_penalty + beta_turn)

    lua_new, n = re.subn(
        pattern,
        rf"\g<1>{new_penalty:.2f}",
        lua,
        count=1,
        flags=re.MULTILINE,
    )

    if n != 1:
        raise RuntimeError(
            f"Expected to replace exactly one standalone turn_penalty, "
            f"but replaced {n}."
        )

    return lua_new, new_penalty


# =========================================================
# Lua calibration
# =========================================================

def calibrate_lua_profile(
    lua: str,
    coef_lookup: dict[str, float],
) -> tuple[str, list[dict]]:
    """
    Apply regression-derived calibration to the Lua profile.
    """

    changes = []

    for coef_name, road_class in ROAD_COEF_MAP.items():
        if coef_name not in coef_lookup:
            continue

        beta = float(coef_lookup[coef_name])

        old_speed = BASE_ROAD_SPEED_KMH[road_class]
        new_speed = convert_speed(old_speed, beta)

        lua = replace_lua_speed(lua, road_class, new_speed)

        was_capped = (
            new_speed == MIN_CALIBRATED_SPEED_KMH
            or new_speed == MAX_CALIBRATED_SPEED_KMH
        )

        changes.append({
            "parameter_type": "road_speed",
            "variable": coef_name,
            "road_class": road_class,
            "coefficient_sec_per_km": beta,
            "old_speed_kmh": old_speed,
            "new_speed_kmh": new_speed,
            "was_capped": was_capped,
        })

    if "n_turns" in coef_lookup:
        beta_turn = float(coef_lookup["n_turns"])

        lua, new_turn_penalty = replace_turn_penalty(
            lua,
            beta_turn,
        )

        changes.append({
            "parameter_type": "turn_penalty",
            "variable": "n_turns",
            "road_class": None,
            "coefficient_sec_per_turn": beta_turn,
            "old_turn_penalty_sec": new_turn_penalty - beta_turn,
            "new_turn_penalty_sec": new_turn_penalty,
            "was_capped": new_turn_penalty == 0,
        })

    return lua, changes


# =========================================================
# Output writing
# =========================================================

def write_outputs(
    lua_original: str,
    lua_calibrated: str,
    changes_df: pl.DataFrame,
) -> None:
    """
    Write calibrated Lua profile and calibration outputs.
    """

    CALIBRATED_LUA.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATED_DIFF.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_CHANGES_CSV.parent.mkdir(parents=True, exist_ok=True)

    CALIBRATED_LUA.write_text(
        lua_calibrated,
        encoding="utf-8",
    )

    diff_text = difflib.unified_diff(
        lua_original.splitlines(keepends=True),
        lua_calibrated.splitlines(keepends=True),
        fromfile=str(ACCESS_LUA),
        tofile=str(CALIBRATED_LUA),
    )

    CALIBRATED_DIFF.write_text(
        "".join(diff_text),
        encoding="utf-8",
    )

    changes_df.write_csv(CALIBRATION_CHANGES_CSV)


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    check_inputs()

    print("Loading regression coefficients...")

    coef_df = pl.read_csv(REGRESSION_COEFFICIENTS_CSV)

    coef_df = filter_coefficients(
        coef_df,
        significant_only=APPLY_ONLY_SIGNIFICANT_COEFS,
        alpha=P_VALUE_THRESHOLD,
    )

    coef_lookup = build_calibration_lookup(coef_df)

    if not coef_lookup:
        raise ValueError("No valid coefficients available for calibration.")

    print("Loading Lua profile...")

    lua_original = ACCESS_LUA.read_text(encoding="utf-8")

    print("Applying calibration...")

    lua_calibrated, changes = calibrate_lua_profile(
        lua_original,
        coef_lookup,
    )

    if not changes:
        raise RuntimeError(
            "No calibration changes were produced from the coefficient table."
        )

    changes_df = pl.DataFrame(changes)

    write_outputs(
        lua_original,
        lua_calibrated,
        changes_df,
    )

    print(f"Saved: {CALIBRATED_LUA}")
    print(f"Saved: {CALIBRATED_DIFF}")
    print(f"Saved: {CALIBRATION_CHANGES_CSV}")


if __name__ == "__main__":
    main()