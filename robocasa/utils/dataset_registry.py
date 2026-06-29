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

    # Composite seen tasks - all 33 tasks from target/composite
    "composite_seen": [
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/ArrangeBreadBasket/20250809/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/ArrangeTea/20250812/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/BreadSelection/20250815/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/CategorizeCondiments/20250814/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/CuttingToolSelection/20250814/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/DeliverStraw/20250813/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/GarnishPancake/20250815/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/GatherTableware/20250815/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/GetToastedBread/20250812/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/HeatKebabSandwich/20250813/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/KettleBoiling/20250814/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/LoadDishwasher/20250811/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/MakeIceLemonade/20250813/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/PackIdenticalLunches/20250815/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/PanTransfer/20250817/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/PortionHotDogs/20250816/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/PreSoakPan/20250809/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/PrepareCoffee/20250812/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/RecycleBottlesByType/20250812/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/RinseSinkBasin/20250816/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/ScrubCuttingBoard/20250816/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/SearingMeat/20250812/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/SeparateFreezerRack/20250815/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/SetUpCuttingStation/20250817/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/StackBowlsCabinet/20250815/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/SteamInMicrowave/20250814/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/StirVegetables/20250814/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/StoreLeftoversInBowl/20250813/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/WaffleReheat/20250817/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/WashFruitColander/20250811/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/WashLettuce/20250814/lerobot", "filter_key": None},
        {"path": "/mnt/ssd_data/mingzhe/robocasa365/datasets_box/v1.0/target/composite/WeighIngredients/20250812/lerobot", "filter_key": None},
    ],
}

# Add all_tasks as a combination after defining the registry
DATASET_SOUP_REGISTRY["all_tasks"] = (
    DATASET_SOUP_REGISTRY["atomic_seen"] +
    DATASET_SOUP_REGISTRY["composite_seen"]
)
