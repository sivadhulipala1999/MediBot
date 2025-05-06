import json


def load_config():
    with open("properties/dev.json") as f:
        return json.load(f)
