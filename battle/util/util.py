import json
import numpy as np
from .types import GameState


class GameHistoryJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GameState):
            return obj._asdict()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)