import json
from enum import Enum
from typing import Annotated, Any, Union, Optional, Dict, List, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator, RootModel

from multiphasessrve.types.array_result import ArrayResult
from multiphasessrve.types.dict_result import DictResult
from multiphasessrve.types.histogram_result import HistogramResult
from multiphasessrve.types.json_results_encoder import ResultJSONEncoder
from multiphasessrve.types.phase import PhasesModel
from multiphasessrve.types.results import AnalysisResult
from multiphasessrve.types.scalar_result import ScalarResult


class OptimizationAlgorithm(str, Enum):
    """Supported optimization algorithms for pymoo."""

    CMAES = "cmaes"
    DE = "de"


class ResultsModel(BaseModel):
    directory: str = Field(..., description="Results output directory")


class AnalysisContextModel(BaseModel):
    """Model for storing analysis-specific context parameters."""

    params: dict[str, Any] = Field(default_factory=dict, description="Analysis-specific parameters")
    model_config = {"arbitrary_types_allowed": True}


class PenaltyModel(BaseModel):
    """Model for defining a single penalty rule"""

    min_value: Optional[float] = Field(None, description="Minimum allowed value")
    max_value: Optional[float] = Field(None, description="Maximum allowed value")
    penalty_factor: float = Field(1.0, description="Multiplier for penalty calculation")
    key: Optional[str] = Field(None, description="Key to apply penalty to (for DictResult)")


# Define weight types
ScalarWeight = Annotated[float, Field(ge=0.0)]
DictWeight = Dict[str, Annotated[float, Field(ge=0.0)]]
Weights = Union[ScalarWeight, DictWeight]


class CoefficientModel(BaseModel):
    name: str
    target: AnalysisResult
    weights: Weights
    penalties: Optional[List[PenaltyModel]] = Field(None, description="List of penalty rules")
    context: AnalysisContextModel = Field(
        default_factory=AnalysisContextModel, description="Analysis-specific context parameters"
    )

    @model_validator(mode="after")
    def validate_penalties(self) -> "CoefficientModel":
        """Validate that penalty configurations match result type"""
        if not self.penalties:
            return self

        if isinstance(self.target, DictResult):
            for penalty in self.penalties:
                if penalty.key and penalty.key not in self.target.values:
                    raise ValueError(f"Penalty applies to non-existent key: {penalty.key}")
        return self

    @model_validator(mode="after")
    def validate_weight_type(self) -> "CoefficientModel":
        """Validate that the weight type matches the target result type."""
        if isinstance(self.target, (ScalarResult, ArrayResult, HistogramResult)):
            if not isinstance(self.weights, (int, float)):
                raise ValueError(f"ScalarResult and ArrayResult require scalar weight, " f"got {type(self.weights)}")
        elif isinstance(self.target, DictResult):
            if not isinstance(self.weights, dict):
                raise ValueError(f"DictResult requires dictionary weight, got {type(self.weights)}")
            # Validate that weight dictionary keys match target keys
            target_keys = set(self.target.values.keys())
            weight_keys = set(self.weights.keys())
            if target_keys != weight_keys:
                missing_keys = target_keys - weight_keys
                extra_keys = weight_keys - target_keys
                error_msg = []
                if missing_keys:
                    error_msg.append(f"Missing weights for keys: {missing_keys}")
                if extra_keys:
                    error_msg.append(f"Extra weight keys not in target: {extra_keys}")
                raise ValueError(". ".join(error_msg))
        return self

    model_config = {"arbitrary_types_allowed": True}


class JobModel(BaseModel):
    number_of_points: Annotated[int, Field(gt=0, description="Number of points for the simulation")]
    element_size: Annotated[int, Field(gt=0, description="Size of elements")]
    threads: Annotated[int, Field(gt=0, description="Number of computation threads")]
    population_size: Annotated[int, Field(gt=0, description="Size of the population")]
    # Optional real size parameter for grain size analysis
    real_size: Optional[Annotated[int, Field(gt=0)]] = Field(None, description="Real size parameter")
    max_generations: Annotated[int, Field(gt=0)] = Field(
        10, description="Maximum number of generations for optimization"
    )
    use_auto_popsize: bool = Field(False, description="Use auto population sizing (CMAES only)")
    fitness_threshold: Optional[Annotated[float, Field(gt=0)]] = Field(
        None, description="Stop optimization when fitness reaches this threshold (lower is better)"
    )
    max_stagnation: Optional[Annotated[int, Field(gt=0)]] = Field(
        None, description="Stop optimization after this many iterations without improvement (pymoo only)"
    )
    cmaes_restarts: Annotated[int, Field(ge=0)] = Field(
        0, description="Number of CMAES restarts after convergence (pymoo only)"
    )
    cmaes_tolfun: Annotated[float, Field(gt=0)] = Field(
        1e-11, description="CMAES termination tolerance for function value changes (lower = stricter)"
    )
    cmaes_tolx: Annotated[float, Field(gt=0)] = Field(
        1e-11, description="CMAES termination tolerance for parameter changes (lower = stricter)"
    )
    cmaes_bipop: bool = Field(
        False, description="Use BIPOP-CMA-ES restart strategy (alternates small/large populations)"
    )
    cmaes_incpopsize: Annotated[int, Field(ge=2)] = Field(
        2, description="Population size multiplier for each restart (2=double, 3=triple)"
    )
    cmaes_sigma: Annotated[float, Field(gt=0, le=1)] = Field(
        0.05, description="Initial step size for CMAES (larger = more exploration)"
    )
    algorithm: OptimizationAlgorithm = Field(
        OptimizationAlgorithm.CMAES, description="Optimization algorithm to use (pymoo only)"
    )
    de_variant: str = Field("DE/best/1/bin", description="DE variant string (pymoo DE only)")
    enable_merging: bool = Field(True, description="Enable grain merging based on compatibility")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility (if None, uses current time)")
    adaptive_weights: bool = Field(False, description="Enable adaptive weight tuning based on metric variance")
    adaptive_weights_calibration_gens: Annotated[int, Field(gt=0)] = Field(
        20, description="Recalibrate and update weights every N generations (scheduled restarts)"
    )
    save_only_final: bool = Field(
        False, description="Save images and pkl files only for the last iteration (best element only)"
    )

    @field_validator("threads")
    def validate_threads(cls, v):
        import multiprocessing

        max_threads = multiprocessing.cpu_count()
        if v > max_threads:
            raise ValueError(f"Number of threads cannot exceed system CPU count ({max_threads})")
        return v

    @model_validator(mode="after")
    def validate_adaptive_weights_interval(self) -> "JobModel":
        """Validate adaptive_weights_calibration_gens against empirical test results.

        WARNING is printed but execution continues (non-blocking validation).
        Based on comprehensive stability tests (20 runs, 4 intervals × 5 seeds):
        - Recommended: interval=10 or interval=20
        - Not recommended: interval>=30 (worse fitness, higher variance)
        """
        if not self.adaptive_weights:
            return self

        interval = self.adaptive_weights_calibration_gens

        if interval < 10 or interval >= 30:
            print("=" * 80)
            print(f"WARNING: adaptive_weights_calibration_gens = {interval}")
            print(f"RECOMMENDED: Use interval 10-20 (empirically tested)")
            print("=" * 80)

        return self


class TaskModel(BaseModel):
    name: Annotated[str, Field(pattern=r"^[A-Z0-9]+$", description="Task identifier")]
    description: Optional[str] = Field(None, description="Optional task description")
    phases: PhasesModel = Field(..., description="Phase definitions")
    job: JobModel
    results: ResultsModel
    coefficients: list[CoefficientModel]

    model_config = {"arbitrary_types_allowed": True}  # To allow custom Result classes

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Custom dump method to handle AnalysisResult serialization"""
        result = super().model_dump(**kwargs)
        # Use the ResultJSONEncoder to convert AnalysisResult objects
        return json.loads(json.dumps(result, cls=ResultJSONEncoder))
