from typing import Union

from multiphasessrve.types.array_result import ArrayResult
from multiphasessrve.types.dict_result import DictResult
from multiphasessrve.types.histogram_result import HistogramResult
from multiphasessrve.types.scalar_result import ScalarResult

# Type alias for all possible result types
AnalysisResult = Union[ScalarResult, ArrayResult, DictResult, HistogramResult]
