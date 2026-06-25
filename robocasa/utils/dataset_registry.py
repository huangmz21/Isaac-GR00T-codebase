"""
Dataset registry for Robocasa datasets.
This file defines dataset soups that can be used for training.
"""

# Registry mapping dataset soup names to lists of dataset configurations
DATASET_SOUP_REGISTRY = {
    # Atomic seen tasks - all 18 tasks from target/atomic
    "atomic_seen": [
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/CloseBlenderLid/20250822/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/CloseFridge/20250816/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/CloseToasterOvenDoor/20250818/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/CoffeeSetupMug/20250813/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/NavigateKitchen/20250821/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/OpenCabinet/20250813/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/OpenDrawer/20250816/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/OpenStandMixerHead/20250818/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/PickPlaceCounterToStove/20250818/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/PickPlaceDrawerToCounter/20250820/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/PickPlaceSinkToCounter/20250813/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/PickPlaceToasterToCounter/20250817/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/SlideDishwasherRack/20250820/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/TurnOffStove/20250812/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/TurnOnElectricKettle/20250817/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/TurnOnMicrowave/20250813/lerobot",
            "filter_key": None,
        },
        {
            "path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/atomic/TurnOnSinkFaucet/20250812/lerobot",
            "filter_key": None,
        },
    ],
}
