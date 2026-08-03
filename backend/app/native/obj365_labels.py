"""Objects365 label table for the D-FINE ``*_obj365`` detector output space.

Ordered ids 0..365 taken verbatim from the onnx-community
``dfine_l_obj365-ONNX`` ``config.json`` ``id2label`` (366 entries — the model
emits ``logits[1, 300, 366]``). Index **0 is a background/"none" placeholder**
(the transformers ``id2label`` reserves it); the 365 real Objects365 categories
occupy ids 1..365. The NMS-free decode argmaxes over all 366 logits, so a
background query can decode as ``class_id 0`` — it resolves to ``"none"`` and is
simply never in any camera's ``detect_objects`` (harmless, filtered downstream).

Names are lowercased with spaces/slashes collapsed to underscores (same
convention as ``coco_labels`` so labels stay URL/UI-safe; ``annotate.plural_label``
renders underscores as spaces): ``"Cabinet/shelf" -> "cabinet_shelf"``,
``"Table Tennis" -> "table_tennis"``.
"""
from __future__ import annotations

OBJ365_LABELS: tuple[str, ...] = (
    "none", "person", "sneakers", "chair", "other_shoes", "hat",
    "car", "lamp", "glasses", "bottle", "desk", "cup",
    "street_lights", "cabinet_shelf", "handbag_satchel", "bracelet", "plate", "picture_frame",
    "helmet", "book", "gloves", "storage_box", "boat", "leather_shoes",
    "flower", "bench", "potted_plant", "bowl_basin", "flag", "pillow",
    "boots", "vase", "microphone", "necklace", "ring", "suv",
    "wine_glass", "belt", "monitor_tv", "backpack", "umbrella", "traffic_light",
    "speaker", "watch", "tie", "trash_bin_can", "slippers", "bicycle",
    "stool", "barrel_bucket", "van", "couch", "sandals", "basket",
    "drum", "pen_pencil", "bus", "wild_bird", "high_heels", "motorcycle",
    "guitar", "carpet", "cell_phone", "bread", "camera", "canned",
    "truck", "traffic_cone", "cymbal", "lifesaver", "towel", "stuffed_toy",
    "candle", "sailboat", "laptop", "awning", "bed", "faucet",
    "tent", "horse", "mirror", "power_outlet", "sink", "apple",
    "air_conditioner", "knife", "hockey_stick", "paddle", "pickup_truck", "fork",
    "traffic_sign", "balloon", "tripod", "dog", "spoon", "clock",
    "pot", "cow", "cake", "dinning_table", "sheep", "hanger",
    "blackboard_whiteboard", "napkin", "other_fish", "orange_tangerine", "toiletry", "keyboard",
    "tomato", "lantern", "machinery_vehicle", "fan", "green_vegetables", "banana",
    "baseball_glove", "airplane", "mouse", "train", "pumpkin", "soccer",
    "skiboard", "luggage", "nightstand", "tea_pot", "telephone", "trolley",
    "head_phone", "sports_car", "stop_sign", "dessert", "scooter", "stroller",
    "crane", "remote", "refrigerator", "oven", "lemon", "duck",
    "baseball_bat", "surveillance_camera", "cat", "jug", "broccoli", "piano",
    "pizza", "elephant", "skateboard", "surfboard", "gun", "skating_and_skiing_shoes",
    "gas_stove", "donut", "bow_tie", "carrot", "toilet", "kite",
    "strawberry", "other_balls", "shovel", "pepper", "computer_box", "toilet_paper",
    "cleaning_products", "chopsticks", "microwave", "pigeon", "baseball", "cutting_chopping_board",
    "coffee_table", "side_table", "scissors", "marker", "pie", "ladder",
    "snowboard", "cookies", "radiator", "fire_hydrant", "basketball", "zebra",
    "grape", "giraffe", "potato", "sausage", "tricycle", "violin",
    "egg", "fire_extinguisher", "candy", "fire_truck", "billiards", "converter",
    "bathtub", "wheelchair", "golf_club", "briefcase", "cucumber", "cigar_cigarette",
    "paint_brush", "pear", "heavy_truck", "hamburger", "extractor", "extension_cord",
    "tong", "tennis_racket", "folder", "american_football", "earphone", "mask",
    "kettle", "tennis", "ship", "swing", "coffee_machine", "slide",
    "carriage", "onion", "green_beans", "projector", "frisbee", "washing_machine_drying_machine",
    "chicken", "printer", "watermelon", "saxophone", "tissue", "toothbrush",
    "ice_cream", "hotair_balloon", "cello", "french_fries", "scale", "trophy",
    "cabbage", "hot_dog", "blender", "peach", "rice", "wallet_purse",
    "volleyball", "deer", "goose", "tape", "tablet", "cosmetics",
    "trumpet", "pineapple", "golf_ball", "ambulance", "parking_meter", "mango",
    "key", "hurdle", "fishing_rod", "medal", "flute", "brush",
    "penguin", "megaphone", "corn", "lettuce", "garlic", "swan",
    "helicopter", "green_onion", "sandwich", "nuts", "speed_limit_sign", "induction_cooker",
    "broom", "trombone", "plum", "rickshaw", "goldfish", "kiwi_fruit",
    "router_modem", "poker_card", "toaster", "shrimp", "sushi", "cheese",
    "notepaper", "cherry", "pliers", "cd", "pasta", "hammer",
    "cue", "avocado", "hamimelon", "flask", "mushroom", "screwdriver",
    "soap", "recorder", "bear", "eggplant", "board_eraser", "coconut",
    "tape_measure_ruler", "pig", "showerhead", "globe", "chips", "steak",
    "crosswalk_sign", "stapler", "camel", "formula_1", "pomegranate", "dishwasher",
    "crab", "hoverboard", "meat_ball", "rice_cooker", "tuba", "calculator",
    "papaya", "antelope", "parrot", "seal", "butterfly", "dumbbell",
    "donkey", "lion", "urinal", "dolphin", "electric_drill", "hair_dryer",
    "egg_tart", "jellyfish", "treadmill", "lighter", "grapefruit", "game_board",
    "mop", "radish", "baozi", "target", "french", "spring_rolls",
    "monkey", "rabbit", "pencil_case", "yak", "red_cabbage", "binoculars",
    "asparagus", "barbell", "scallop", "noddles", "comb", "dumpling",
    "oyster", "table_tennis_paddle", "cosmetics_brush_eyeliner_pencil", "chainsaw", "eraser", "lobster",
    "durian", "okra", "lipstick", "cosmetics_mirror", "curling", "table_tennis",
)

assert len(OBJ365_LABELS) == 366, len(OBJ365_LABELS)
