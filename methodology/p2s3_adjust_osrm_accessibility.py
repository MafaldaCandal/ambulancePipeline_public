"""
p2s3_adjust_osrm_accessibility.py

Phase 2, step 3.

Purpose:
    Create a priority-vehicle OSRM access profile for A0/A1 ambulance
    deployments.

Inputs:
    - Active OSM .pbf file, resolved by paths.OSM_PBF
    - Default OSRM car.lua, extracted from the configured Docker image

Outputs:
    - runs/RunXXX/ambulance_nl.lua
    - runs/RunXXX/ambulance_profile.diff

Methodological interpretation:
    This profile is a legally motivated operational approximation for urgent
    ambulance trips using optical and sound signals. It softens access
    restrictions that priority vehicles may plausibly override, while retaining
    the physical structure of the road network and preserving OSRM turn
    restrictions.

    It is not a complete legal simulation. Static OSRM profiles cannot fully
    represent context-dependent rules such as cautious red-light passage,
    emergency-lane use under traffic conditions, or driver judgement.
"""

from __future__ import annotations

import difflib
import re

from paths import (
    ORIGINAL_LUA,
    OSM_PBF,
    ACCESS_LUA,
    ACCESS_DIFF,
)

from utils.osrm_utils import extract_default_car_lua


# =========================================================
# Input handling
# =========================================================

def create_original_lua() -> None:
    """
    Extract the default OSRM car profile.

    The extracted profile is used as the reproducible baseline from which the
    ambulance priority-access profile is derived.
    """
    print("Extracting OSRM default car.lua from Docker image...")
    extract_default_car_lua(ORIGINAL_LUA)


def check_inputs() -> None:
    """
    Validate required inputs and create the default Lua profile if needed.
    """
    if not OSM_PBF.exists():
        raise FileNotFoundError(f"Missing OSM file: {OSM_PBF}")

    create_original_lua()

    if not ORIGINAL_LUA.exists():
        raise FileNotFoundError(
            f"Could not create default OSRM car.lua at: {ORIGINAL_LUA}"
        )

    print("Input files found.")


# =========================================================
# Lua editing helpers
# =========================================================

def replace_lua_block(lua: str, name: str, replacement: str) -> str:
    """
    Replace one named Lua Set or Sequence block.

    Strict replacement is intentional: if the upstream OSRM profile structure
    changes, this script should fail rather than silently producing an invalid
    profile.
    """
    pattern = rf"{name}\s*=\s*(Set|Sequence)\s*\{{.*?\n\s*\}}"

    new_lua, n = re.subn(
        pattern,
        replacement,
        lua,
        count=1,
        flags=re.DOTALL,
    )

    if n != 1:
        raise RuntimeError(
            f"Expected to replace exactly one Lua block named {name!r}, "
            f"but replaced {n}."
        )

    return new_lua


def ensure_turn_restrictions_enabled(lua: str) -> str:
    """
    Ensure OSRM turn restrictions remain enabled.

    Priority-vehicle privileges do not imply that all prohibited turns or
    opposite-direction movements should become normal route choices. Keeping
    turn restrictions enabled makes the access profile conservative.
    """
    pattern = r"(use_turn_restrictions\s*=\s*)(true|false)"

    new_lua, n = re.subn(
        pattern,
        r"\g<1>true",
        lua,
        count=1,
    )

    if n != 1:
        raise RuntimeError(
            "Expected to find exactly one use_turn_restrictions setting "
            f"in the Lua profile, but found/replaced {n}."
        )

    return new_lua


# =========================================================
# Priority-vehicle access profile
# =========================================================

def create_ambulance_profile() -> None:
    """
    Create the ambulance priority-access OSRM profile.

    Access restrictions that plausibly correspond to legal access limitations
    are softened. Physical barriers are treated more conservatively.
    """
    lua_original = ORIGINAL_LUA.read_text(encoding="utf-8")
    lua = lua_original

    replacements = {
        "access_tag_whitelist": """access_tag_whitelist = Set {
    'yes', 'motorcar', 'motor_vehicle', 'vehicle',
    'permissive', 'designated',
    'destination', 'delivery', 'customers', 'private',
    'official', 'emergency', 'psv', 'bus',
  }""",

        "access_tag_blacklist": """access_tag_blacklist = Set {
    'no', 'agricultural', 'forestry', 'use_sidepath', 'dismount',
  }""",

        "service_access_tag_blacklist": """service_access_tag_blacklist = Set {
  }""",

        "service_tag_forbidden": """service_tag_forbidden = Set {
  }""",

        # Conservative barrier treatment:
        # - gates and lift gates are often controllable or passable in
        #   emergency contexts;
        # - fixed physical restrictions such as height restrictors are not
        #   automatically whitelisted.
        "barrier_whitelist": """barrier_whitelist = Set {
    'cattle_grid', 'border_control', 'toll_booth',
    'sally_port', 'gate', 'lift_gate',
    'entrance',
  }""",
    }

    for block_name, replacement in replacements.items():
        lua = replace_lua_block(lua, block_name, replacement)

    lua = ensure_turn_restrictions_enabled(lua)

    ACCESS_LUA.write_text(lua, encoding="utf-8")

    diff_text = difflib.unified_diff(
        lua_original.splitlines(keepends=True),
        lua.splitlines(keepends=True),
        fromfile="car.lua",
        tofile="ambulance_nl.lua",
    )

    ACCESS_DIFF.write_text("".join(diff_text), encoding="utf-8")

    print(f"Created {ACCESS_LUA}")
    print(f"Created {ACCESS_DIFF}")


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    check_inputs()
    create_ambulance_profile()
    print("Priority-vehicle access profile created.")


if __name__ == "__main__":
    main()