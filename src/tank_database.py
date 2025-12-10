import json
import os

class TankDatabase:
    def __init__(self, json_path="../data/tanks.json"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"No se encontró el archivo JSON en {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def find_tank(self, label):
        """
        Busca el tanque por nombre en todas las naciones.
        label = nombre detectado por YOLO (ej: 'Pz. III', 'T-34', 'Chi-Nu')
        """
        for nation, tanks in self.data.items():
            for tank_name, tank_info in tanks.items():
                if tank_name.lower() == label.lower():
                    return {
                        "nation": nation,
                        "name": tank_name,
                        **tank_info
                    }
        return None
