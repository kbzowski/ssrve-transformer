from json import JSONEncoder

import numpy as np

from multiphasessrve.types.results import ScalarResult, ArrayResult, DictResult


class ResultJSONEncoder(JSONEncoder):
    """Custom JSON encoder for Result classes."""

    def default(self, obj):
        if isinstance(obj, (ScalarResult, ArrayResult, DictResult)):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
