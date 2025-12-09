import json

class TankDatabase:
    def __init__(self, file_path="data/tanks.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

    def exists(self, name):
        return name in self.db

    def get(self, name):
        return self.db.get(name, None)
