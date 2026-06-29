"""
Dataset registry for Robocasa datasets.
This file defines dataset soups that can be used for training.
"""

# Base directory for atomic target datasets
_ATOMIC_BASE = "/data1/robocasa365/datasets_box/v1.0/target/atomic"

# All 18 atomic_seen tasks: task name -> dataset date folder.
# The single source of truth for both the joint soup and the per-task soups.
ATOMIC_SEEN_TASKS = {
    "CloseBlenderLid": "20250822",
    "CloseFridge": "20250816",
    "CloseToasterOvenDoor": "20250818",
    "CoffeeSetupMug": "20250813",
    "NavigateKitchen": "20250821",
    "OpenCabinet": "20250813",
    "OpenDrawer": "20250816",
    "OpenStandMixerHead": "20250818",
    "PickPlaceCounterToCabinet": "20250811",
    "PickPlaceCounterToStove": "20250818",
    "PickPlaceDrawerToCounter": "20250820",
    "PickPlaceSinkToCounter": "20250813",
    "PickPlaceToasterToCounter": "20250817",
    "SlideDishwasherRack": "20250820",
    "TurnOffStove": "20250812",
    "TurnOnElectricKettle": "20250817",
    "TurnOnMicrowave": "20250813",
    "TurnOnSinkFaucet": "20250812",
}


def _task_entry(task_name: str) -> dict:
    """Build a single dataset config entry for a given atomic task."""
    date = ATOMIC_SEEN_TASKS[task_name]
    return {
        "path": f"{_ATOMIC_BASE}/{task_name}/{date}/lerobot",
        "filter_key": None,
    }


# Registry mapping dataset soup names to lists of dataset configurations
DATASET_SOUP_REGISTRY = {
    # Atomic seen tasks - all 18 tasks from target/atomic
    "atomic_seen": [_task_entry(t) for t in ATOMIC_SEEN_TASKS],
}

# Auto-register a single-task soup for each of the 18 atomic tasks.
# Soup key is the lowercased task name + "_only", e.g. "opencabinet_only".
for _task in ATOMIC_SEEN_TASKS:
    DATASET_SOUP_REGISTRY[f"{_task.lower()}_only"] = [_task_entry(_task)]

# Manual combinations
DATASET_SOUP_REGISTRY["opencabinet_opendrawer"] = [
    _task_entry("OpenCabinet"),
    _task_entry("OpenDrawer"),
]
