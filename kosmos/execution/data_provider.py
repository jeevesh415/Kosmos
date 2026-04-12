"""
Data provisioning for experiment execution.

Issue #51 fix: Provides synthetic data generation for computational experiments
to prevent infinite loops when data files are missing.

Supports:
1. Synthetic data generation based on experiment type and domain
2. Data validation before execution
3. Fallback mechanisms for missing files
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pathlib import Path
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataSourceType(str, Enum):
    """Types of data sources for experiments."""
    FILE = "file"              # Load from file path
    SYNTHETIC = "synthetic"    # Generate synthetic data
    INLINE = "inline"          # Data provided inline in protocol


class SyntheticDataGenerator:
    """
    Generates synthetic data based on experiment specifications.

    Issue #51: This enables computational experiments to run without
    requiring external data files, preventing infinite loops.
    """

    # Domain-specific data templates
    DOMAIN_TEMPLATES = {
        "biology": {
            "ttest_comparison": {
                "columns": {"group": "categorical", "measurement": "continuous"},
                "groups": ["control", "treatment"],
                "effect_size": 0.5,
                "n_per_group": 50
            },
            "gene_expression": {
                "columns": {"gene_id": "identifier", "expression": "continuous", "condition": "categorical"},
                "conditions": ["baseline", "treated"],
                "n_genes": 100
            },
            "metabolomics": {
                "columns": {"metabolite": "identifier", "concentration": "continuous", "sample": "categorical"},
                "effect_size": 0.3
            }
        },
        "neuroscience": {
            "ttest_comparison": {
                "columns": {"group": "categorical", "measurement": "continuous"},
                "groups": ["control", "experimental"],
                "effect_size": 0.4,
                "n_per_group": 30
            },
            "connectivity": {
                "columns": {"source": "identifier", "target": "identifier", "weight": "continuous"},
                "n_nodes": 50,
                "density": 0.1
            }
        },
        "physics": {
            "correlation_analysis": {
                "columns": {"x": "continuous", "y": "continuous"},
                "correlation": 0.7,
                "n_samples": 100
            },
            "parameter_sweep": {
                "columns": {"parameter": "continuous", "output": "continuous"},
                "relationship": "linear",
                "noise_level": 0.1
            }
        },
        "statistics": {
            "ttest_comparison": {
                "columns": {"group": "categorical", "measurement": "continuous"},
                "groups": ["control", "experimental"],
                "effect_size": 0.5,
                "n_per_group": 50
            },
            "correlation_analysis": {
                "columns": {"x": "continuous", "y": "continuous"},
                "correlation": 0.7,
                "n_samples": 100
            },
            "anova": {
                "columns": {"group": "categorical", "measurement": "continuous"},
                "groups": ["A", "B", "C"],
                "effect_size": 0.4,
                "n_per_group": 30
            }
        },
        "machine_learning": {
            "classification": {
                "columns": {"feature_1": "continuous", "feature_2": "continuous", "label": "categorical"},
                "classes": ["class_0", "class_1"],
                "n_samples": 200,
                "separability": 0.8
            },
            "regression": {
                "columns": {"x": "continuous", "y": "continuous"},
                "relationship": "linear",
                "noise_level": 0.2,
                "n_samples": 150
            }
        }
    }

    def __init__(self, seed: int = 42):
        """
        Initialize generator with random seed for reproducibility.

        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
        self.seed = seed

    def generate(
        self,
        domain: Optional[str] = None,
        experiment_type: Optional[str] = None,
        n_samples: int = 100,
        columns: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data based on specification.

        Args:
            domain: Scientific domain (biology, physics, etc.)
            experiment_type: Type of experiment (ttest_comparison, correlation_analysis, etc.)
            n_samples: Number of samples to generate
            columns: Optional column specifications
            **kwargs: Additional parameters (effect_size, correlation, etc.)

        Returns:
            pd.DataFrame: Generated synthetic data
        """
        # Try domain-specific template first
        template = self._get_template(domain, experiment_type)

        if template:
            return self._generate_from_template(template, n_samples, **kwargs)

        # Fall back to generic generation based on columns
        if columns:
            return self._generate_from_columns(columns, n_samples, **kwargs)

        # Default: generate basic two-group comparison data
        return self._generate_default_data(n_samples, **kwargs)

    def _get_template(
        self,
        domain: Optional[str],
        experiment_type: Optional[str]
    ) -> Optional[Dict]:
        """Get domain-specific template if available."""
        if not domain or not experiment_type:
            return None

        domain_key = domain.lower().replace(" ", "_")
        exp_key = experiment_type.lower().replace(" ", "_").replace("-", "_")

        domain_templates = self.DOMAIN_TEMPLATES.get(domain_key, {})
        return domain_templates.get(exp_key)

    def _generate_from_template(
        self,
        template: Dict,
        n_samples: int,
        **kwargs
    ) -> pd.DataFrame:
        """Generate data using domain-specific template."""
        columns = template.get("columns", {})
        effect_size = kwargs.get("effect_size")
        if effect_size is None:
            effect_size = self._randomize_effect_size()
        groups = kwargs.get("groups", template.get("groups", ["control", "experimental"]))
        n_per_group = kwargs.get("n_per_group", template.get("n_per_group", n_samples // len(groups)))

        data = {}
        n_total = n_per_group * len(groups)

        for col_name, col_type in columns.items():
            if col_type == "categorical":
                # Assign groups evenly
                data[col_name] = np.repeat(groups, n_per_group)
            elif col_type == "continuous":
                # Generate with group effect if this is a measurement column
                values = []
                for i, group in enumerate(groups):
                    # First group is baseline (mean=0), others have effect
                    mean = effect_size * i
                    values.extend(self.rng.normal(mean, 1.0, n_per_group))
                data[col_name] = values
            elif col_type == "identifier":
                data[col_name] = [f"ID_{i}" for i in range(n_total)]

        df = pd.DataFrame(data)

        # Shuffle to avoid ordering effects
        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def _generate_from_columns(
        self,
        columns: Dict[str, str],
        n_samples: int,
        **kwargs
    ) -> pd.DataFrame:
        """Generate data from column specifications."""
        data = {}

        for col_name, col_type in columns.items():
            if col_type in ["categorical", "group"]:
                groups = kwargs.get("groups", ["A", "B"])
                data[col_name] = self.rng.choice(groups, n_samples)
            elif col_type in ["continuous", "numeric", "float"]:
                data[col_name] = self.rng.normal(0, 1, n_samples)
            elif col_type in ["integer", "count"]:
                data[col_name] = self.rng.randint(0, 100, n_samples)
            elif col_type in ["identifier", "id"]:
                data[col_name] = [f"ID_{i}" for i in range(n_samples)]
            else:
                # Default to continuous
                data[col_name] = self.rng.normal(0, 1, n_samples)

        return pd.DataFrame(data)

    def _randomize_effect_size(self) -> float:
        """Randomize effect size to prevent guaranteed false positives.

        Distribution: 30% null (0.0), 20% small (0.2), 20% medium (0.5),
        20% large (0.8), 10% random uniform [0, 1].
        """
        roll = self.rng.random()
        if roll < 0.30:
            return 0.0
        elif roll < 0.50:
            return 0.2
        elif roll < 0.70:
            return 0.5
        elif roll < 0.90:
            return 0.8
        else:
            return float(self.rng.uniform(0.0, 1.0))

    def _generate_default_data(
        self,
        n_samples: int,
        **kwargs
    ) -> pd.DataFrame:
        """Generate default two-group comparison data."""
        effect_size = kwargs.get("effect_size")
        if effect_size is None:
            effect_size = self._randomize_effect_size()
        n_per_group = n_samples // 2

        # Generate control and experimental groups
        control = self.rng.normal(0, 1, n_per_group)
        experimental = self.rng.normal(effect_size, 1, n_per_group)

        df = pd.DataFrame({
            "group": ["control"] * n_per_group + ["experimental"] * n_per_group,
            "measurement": np.concatenate([control, experimental])
        })

        return df.sample(frac=1, random_state=self.seed).reset_index(drop=True)


class DataProvider:
    """
    Main interface for providing data to experiments.

    Issue #51 fix: Provides mechanisms to:
    1. Validate data availability before execution
    2. Fall back to synthetic data when files are missing
    3. Generate domain-appropriate synthetic data

    Usage:
        provider = DataProvider()

        # Check if data is available
        available, msg = provider.validate_data(spec)

        # Get data with synthetic fallback
        df, source = provider.get_data(spec, allow_synthetic=True)
    """

    def __init__(self, default_data_dir: Optional[str] = None, seed: int = 42):
        """
        Initialize data provider.

        Args:
            default_data_dir: Default directory for data files
            seed: Random seed for synthetic data generation
        """
        self.default_data_dir = Path(default_data_dir) if default_data_dir else None
        self.generator = SyntheticDataGenerator(seed=seed)

    def get_data(
        self,
        file_path: Optional[str] = None,
        domain: Optional[str] = None,
        experiment_type: Optional[str] = None,
        n_samples: int = 100,
        allow_synthetic: bool = True,
        **kwargs
    ) -> Tuple[pd.DataFrame, str]:
        """
        Get data for experiment execution.

        Args:
            file_path: Optional path to data file
            domain: Scientific domain
            experiment_type: Type of experiment
            n_samples: Number of samples for synthetic data
            allow_synthetic: Whether to allow synthetic data fallback
            **kwargs: Additional parameters for data generation

        Returns:
            Tuple of (DataFrame, source_description)

        Raises:
            FileNotFoundError: If file not found and synthetic not allowed
        """
        # Try to load from file first
        if file_path:
            path = Path(file_path).resolve()

            # Validate file extension
            _ALLOWED_EXTENSIONS = {'.csv', '.tsv', '.parquet', '.json', '.jsonl', '.xlsx'}
            if path.suffix.lower() not in _ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"Disallowed file extension '{path.suffix}'. "
                    f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
                )

            # Path traversal check: ensure resolved path is under allowed directory
            if self.default_data_dir is not None:
                allowed_root = self.default_data_dir.resolve()
                try:
                    path.relative_to(allowed_root)
                except ValueError:
                    raise ValueError(
                        f"Path '{path}' is outside the allowed data directory '{allowed_root}'"
                    )

            if path.exists():
                try:
                    suffix = path.suffix.lower()
                    if suffix == '.tsv':
                        df = pd.read_csv(path, sep='\t')
                    elif suffix == '.parquet':
                        df = pd.read_parquet(path)
                    elif suffix == '.json':
                        df = pd.read_json(path)
                    elif suffix == '.jsonl':
                        df = pd.read_json(path, lines=True)
                    elif suffix == '.xlsx':
                        df = pd.read_excel(path)
                    else:
                        df = pd.read_csv(path)
                    logger.info(f"Loaded data from file: {file_path} ({len(df)} rows)")
                    return df, f"file:{file_path}"
                except Exception as e:
                    logger.warning(f"Failed to load file {file_path}: {e}")
                    if not allow_synthetic:
                        raise
            else:
                logger.warning(f"Data file not found: {file_path}")
                if not allow_synthetic:
                    raise FileNotFoundError(f"Data file not found: {file_path}")

        # Generate synthetic data
        if allow_synthetic:
            df = self.generator.generate(
                domain=domain,
                experiment_type=experiment_type,
                n_samples=n_samples,
                **kwargs
            )
            source = f"synthetic:{domain or 'general'}/{experiment_type or 'default'}"
            logger.info(f"Generated synthetic data: {source} ({len(df)} rows)")
            return df, source

        raise ValueError("No data source available and synthetic generation not allowed")

    def validate_data(
        self,
        file_path: Optional[str] = None,
        allow_synthetic: bool = True
    ) -> Tuple[bool, str]:
        """
        Check if data is available without loading it.

        Args:
            file_path: Path to data file
            allow_synthetic: Whether synthetic fallback is allowed

        Returns:
            Tuple of (is_available, message)
        """
        if file_path and Path(file_path).exists():
            return True, f"File exists: {file_path}"

        if allow_synthetic:
            return True, "Synthetic data will be generated"

        return False, f"File not found and synthetic not allowed: {file_path}"


def generate_inline_data_code(
    domain: Optional[str] = None,
    experiment_type: Optional[str] = None,
    n_samples: int = 100,
    group_var: str = "group",
    measure_var: str = "measurement",
    groups: Optional[List[str]] = None,
    effect_size: float = 0.5,
    seed: int = 42
) -> str:
    """
    Generate Python code that creates synthetic data inline.

    Issue #51: This allows code templates to include data generation
    instead of requiring external files.

    Args:
        domain: Scientific domain
        experiment_type: Type of experiment
        n_samples: Number of samples
        group_var: Name of group column
        measure_var: Name of measurement column
        groups: List of group names
        effect_size: Effect size between groups
        seed: Random seed

    Returns:
        Python code string that generates a DataFrame
    """
    groups = groups or ["control", "experimental"]
    n_per_group = n_samples // len(groups)

    # Validate variable names are safe Python identifiers to prevent code injection
    if not group_var.isidentifier():
        raise ValueError(
            f"group_var '{group_var}' is not a valid Python identifier"
        )
    if not measure_var.isidentifier():
        raise ValueError(
            f"measure_var '{measure_var}' is not a valid Python identifier"
        )

    code = f'''# Synthetic data generation (Issue #51 fix)
# This enables computational experiments without external data files
import numpy as np
import pandas as pd

np.random.seed({seed})
n_per_group = {n_per_group}
effect_size = {effect_size}
groups = {groups}

# Generate synthetic data with expected effect size
data = []
for i, group in enumerate(groups):
    mean = effect_size * i  # Control at 0, treatment shifted
    values = np.random.normal(mean, 1.0, n_per_group)
    for v in values:
        data.append({{"{group_var}": group, "{measure_var}": v}})

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state={seed}).reset_index(drop=True)
'''

    return code
