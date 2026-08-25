def _teleport(x, y, z, yaw):
    return {"action": "TeleportFull", "position": {"x": x, "y": y, "z": z}, "rotation": {"x": 0, "y": yaw, "z": 0}, "horizon": 30, "standing": True}


LOCATIONS = {
    "FloorPlan2": {
        "Fridge": _teleport(-0.75, 0.91, 0.0, 270),
        "Sink": _teleport(-0.75, 0.91, -0.5, 180),
        "Table 1": _teleport(0.0, 0.91, -0.75, 0),
        "Microwave": _teleport(0.85, 0.91, -0.35, 150),
        "Stove": _teleport(0.75, 0.91, 0.75, 90),
        "Table 2": _teleport(1.0, 0.91, 1.25, 270),
        "Door": _teleport(-0.75, 0.91, 3.0, 180),
    },
}

EXCLUDE = {
    "FloorPlan2": {
        "Fridge": ["Cabinet 2"],
        "Sink": ["Drawer 1", "Window 2"],
        "Table 1": ["Fridge 1", "Bowl 1", "CellPhone 1", "Drawer 4", "Drawer 8", "Drawer 9", "Drawer 10", "CounterTop 1"],
        "Microwave": ["Drawer 3", "Cabinet 5", "Cabinet 6"],
        "Stove": ["Window 3", "Microwave 1", "StoveKnob 1", "StoveKnob 2", "StoveKnob 3", "StoveKnob 4", "StoveKnob 5", "StoveKnob 6", "StoveBurner 1", "StoveBurner 5"],
        "Table 2": ["CounterTop 1"],
        "Door": ["CellPhone 1", "Potato 1", "Chair 1", "GarbageCan 1", "CounterTop 2"],
    },
}
