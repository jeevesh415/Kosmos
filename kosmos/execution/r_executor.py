"""
R language execution engine.

Executes R code safely with output capture, error handling, and result parsing.
Supports both direct Rscript execution and Docker-based sandboxed execution.
"""

import subprocess
import tempfile
import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RExecutionResult:
    """Result of R code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time: float = 0.0
    exit_code: Optional[int] = None
    output_files: List[str] = field(default_factory=list)
    parsed_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'return_value': self.return_value,
            'error': self.error,
            'error_type': self.error_type,
            'execution_time': self.execution_time,
            'exit_code': self.exit_code,
            'output_files': self.output_files,
            'parsed_results': self.parsed_results
        }


class RExecutor:
    """
    Executes R code with output capture and result parsing.

    Provides:
    - Rscript-based execution
    - Output capture (stdout/stderr)
    - Structured result extraction
    - Support for statistical genetics packages (TwoSampleMR, susieR)
    - Docker sandbox integration
    """

    # R script wrapper template for result capture
    WRAPPER_TEMPLATE = '''
# Wrapper to capture results as JSON
.kosmos_results <- list()

# Helper function to capture results
kosmos_capture <- function(name, value) {{
    .kosmos_results[[name]] <<- value
}}

# Run user code
tryCatch({{
{user_code}
}}, error = function(e) {{
    cat("R_ERROR:", conditionMessage(e), "\\n", file=stderr())
}})

# Output captured results as JSON
if (length(.kosmos_results) > 0) {{
    cat("KOSMOS_RESULTS_START\\n")
    cat(jsonlite::toJSON(.kosmos_results, auto_unbox=TRUE, pretty=TRUE))
    cat("\\nKOSMOS_RESULTS_END\\n")
}}
'''

    def __init__(
        self,
        r_path: str = "Rscript",
        timeout: int = 300,
        working_dir: Optional[str] = None,
        use_docker: bool = False,
        docker_image: str = "kosmos-sandbox-r:latest"
    ):
        """
        Initialize R executor.

        Args:
            r_path: Path to Rscript executable
            timeout: Maximum execution time in seconds
            working_dir: Working directory for R execution
            use_docker: If True, use Docker sandbox for execution
            docker_image: Docker image name for R execution
        """
        self.r_path = r_path
        self.timeout = timeout
        self.working_dir = working_dir
        self.use_docker = use_docker
        self.docker_image = docker_image

    def is_r_available(self) -> bool:
        """Check if R is available on the system."""
        try:
            result = subprocess.run(
                [self.r_path, "--version"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_r_version(self) -> Optional[str]:
        """Get R version string."""
        try:
            result = subprocess.run(
                [self.r_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                # Parse version from output
                match = re.search(r'R version (\d+\.\d+\.\d+)', result.stdout)
                if match:
                    return match.group(1)
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    def detect_language(self, code: str) -> str:
        """
        Detect if code is R or Python.

        Args:
            code: Code string to analyze

        Returns:
            'r' or 'python'
        """
        # Check for shebang
        if code.strip().startswith('#!/usr/bin/env Rscript') or \
           code.strip().startswith('#!/usr/bin/Rscript'):
            return 'r'

        # R-specific patterns
        r_patterns = [
            r'\b(library|require)\s*\(',  # library() or require()
            r'<-\s*function',  # function assignment
            r'\bc\s*\([^)]+\)',  # c() vector
            r'\bdata\.frame\s*\(',  # data.frame()
            r'\bggplot\s*\(',  # ggplot
            r'\bdplyr::|tidyr::',  # tidyverse namespaces
            r'\bTwoSampleMR::|MendelianRandomization::',  # MR packages
            r'%>%|%\+%',  # pipe operators
            r'\bNA\b(?!me)',  # NA literal (not NAme, etc.)
            r'\bNULL\b',  # NULL literal
            r'<-(?!=)',  # assignment operator (not <-=)
        ]

        # Python-specific patterns
        python_patterns = [
            r'\bimport\s+\w+',  # import statement
            r'\bfrom\s+\w+\s+import',  # from import
            r'\bdef\s+\w+\s*\(',  # function definition
            r'\bclass\s+\w+[:\(]',  # class definition
            r'\bif\s+__name__\s*==',  # main guard
            r'\bprint\s*\([^)]*\)',  # print function
            r'^\s*#\s*-\*-.*python',  # python encoding declaration
        ]

        r_score = sum(1 for p in r_patterns if re.search(p, code, re.MULTILINE))
        python_score = sum(1 for p in python_patterns if re.search(p, code, re.MULTILINE))

        if r_score > python_score:
            return 'r'
        elif python_score > r_score:
            return 'python'
        else:
            # Default to Python if unclear
            return 'python'

    def execute(
        self,
        code: str,
        capture_results: bool = True,
        output_dir: Optional[str] = None
    ) -> RExecutionResult:
        """
        Execute R code and capture results.

        Args:
            code: R code to execute
            capture_results: If True, wrap code to capture results as JSON
            output_dir: Directory for output files

        Returns:
            RExecutionResult with output and parsed results
        """
        start_time = datetime.now()

        # Create temp directory for execution
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = output_dir or self.working_dir or temp_dir

            # Prepare code with result capture wrapper if requested
            if capture_results:
                # Indent user code for wrapper
                indented_code = '\n'.join('    ' + line for line in code.split('\n'))
                wrapped_code = self.WRAPPER_TEMPLATE.format(user_code=indented_code)
                # Add jsonlite dependency
                wrapped_code = "if (!require('jsonlite', quietly=TRUE)) install.packages('jsonlite', repos='https://cloud.r-project.org', quiet=TRUE)\n" + wrapped_code
            else:
                wrapped_code = code

            # Write code to temp file
            script_path = Path(temp_dir) / "script.R"
            script_path.write_text(wrapped_code)

            try:
                if self.use_docker:
                    result = self._execute_docker(script_path, work_dir)
                else:
                    result = self._execute_local(script_path, work_dir)

                execution_time = (datetime.now() - start_time).total_seconds()
                result.execution_time = execution_time

                # Parse captured results from stdout
                if capture_results and result.success:
                    result.parsed_results = self._parse_results(result.stdout)

                # Collect output files
                if output_dir:
                    result.output_files = self._collect_output_files(output_dir)

                return result

            except subprocess.TimeoutExpired:
                execution_time = (datetime.now() - start_time).total_seconds()
                return RExecutionResult(
                    success=False,
                    error="Execution timed out",
                    error_type="TimeoutError",
                    execution_time=execution_time
                )
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                return RExecutionResult(
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                    execution_time=execution_time
                )

    def _execute_local(self, script_path: Path, work_dir: str) -> RExecutionResult:
        """Execute R script locally using Rscript."""
        result = subprocess.run(
            [self.r_path, str(script_path)],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=work_dir
        )

        # Check for R_ERROR in stderr
        error = None
        error_type = None
        if "R_ERROR:" in result.stderr:
            match = re.search(r'R_ERROR:\s*(.+)', result.stderr)
            if match:
                error = match.group(1).strip()
                error_type = "RError"

        success = result.returncode == 0 and error is None

        return RExecutionResult(
            success=success,
            stdout=result.stdout,
            stderr=result.stderr,
            error=error,
            error_type=error_type,
            exit_code=result.returncode
        )

    def _execute_docker(self, script_path: Path, work_dir: str) -> RExecutionResult:
        """Execute R script in Docker container."""
        try:
            import docker
        except ImportError:
            raise ImportError("docker package required for Docker execution")

        client = docker.from_env()

        # Read script content
        script_content = script_path.read_text()

        # Create container
        container = client.containers.run(
            self.docker_image,
            command=["Rscript", "-e", script_content],
            volumes={work_dir: {'bind': '/workspace', 'mode': 'rw'}},
            working_dir='/workspace',
            network_disabled=True,
            mem_limit='2g',
            cpu_period=100000,
            cpu_quota=200000,  # 2 CPUs
            detach=True,
            remove=False
        )

        try:
            # Wait for completion
            result = container.wait(timeout=self.timeout)
            exit_code = result.get('StatusCode', -1)

            # Get logs
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')

            # Check for errors
            error = None
            error_type = None
            if "R_ERROR:" in stderr:
                match = re.search(r'R_ERROR:\s*(.+)', stderr)
                if match:
                    error = match.group(1).strip()
                    error_type = "RError"

            success = exit_code == 0 and error is None

            return RExecutionResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                error=error,
                error_type=error_type,
                exit_code=exit_code
            )

        finally:
            # Clean up container
            try:
                container.remove(force=True)
            except Exception as e:
                logger.debug(f"R executor container cleanup failed: {e}")

    def _parse_results(self, stdout: str) -> Dict[str, Any]:
        """Parse JSON results from R output."""
        results = {}

        # Look for our result markers
        match = re.search(
            r'KOSMOS_RESULTS_START\s*\n(.+?)\nKOSMOS_RESULTS_END',
            stdout,
            re.DOTALL
        )

        if match:
            try:
                json_str = match.group(1).strip()
                results = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse R results JSON: {e}")

        return results

    def _collect_output_files(self, output_dir: str) -> List[str]:
        """Collect paths to output files generated by R script."""
        output_files = []
        output_path = Path(output_dir)

        # Common output file extensions
        output_extensions = {'.png', '.pdf', '.svg', '.csv', '.tsv', '.rds', '.rdata', '.json'}

        for file_path in output_path.iterdir():
            if file_path.suffix.lower() in output_extensions:
                output_files.append(str(file_path))

        return output_files

    def execute_mendelian_randomization(
        self,
        exposure_data: Dict[str, Any],
        outcome_data: Dict[str, Any],
        method: str = "mr_ivw"
    ) -> RExecutionResult:
        """
        Execute Mendelian Randomization analysis using TwoSampleMR.

        This is a convenience method for running MR analyses as described
        in the Kosmos paper for statistical genetics discoveries.

        Args:
            exposure_data: Dictionary with exposure GWAS summary statistics
            outcome_data: Dictionary with outcome GWAS summary statistics
            method: MR method to use (mr_ivw, mr_egger, mr_weighted_median, etc.)

        Returns:
            RExecutionResult with MR results
        """
        # Generate R code for MR analysis
        code = f'''
library(TwoSampleMR)

# Load exposure data
exposure_dat <- data.frame(
    SNP = c({', '.join(f'"{s}"' for s in exposure_data.get('snp', []))}),
    beta = c({', '.join(str(b) for b in exposure_data.get('beta', []))}),
    se = c({', '.join(str(s) for s in exposure_data.get('se', []))}),
    effect_allele = c({', '.join(f'"{a}"' for a in exposure_data.get('effect_allele', []))}),
    other_allele = c({', '.join(f'"{a}"' for a in exposure_data.get('other_allele', []))}),
    pval = c({', '.join(str(p) for p in exposure_data.get('pval', []))})
)
exposure_dat$exposure <- "{exposure_data.get('exposure_name', 'exposure')}"

# Load outcome data
outcome_dat <- data.frame(
    SNP = c({', '.join(f'"{s}"' for s in outcome_data.get('snp', []))}),
    beta = c({', '.join(str(b) for b in outcome_data.get('beta', []))}),
    se = c({', '.join(str(s) for s in outcome_data.get('se', []))}),
    effect_allele = c({', '.join(f'"{a}"' for a in outcome_data.get('effect_allele', []))}),
    other_allele = c({', '.join(f'"{a}"' for a in outcome_data.get('other_allele', []))}),
    pval = c({', '.join(str(p) for p in outcome_data.get('pval', []))})
)
outcome_dat$outcome <- "{outcome_data.get('outcome_name', 'outcome')}"

# Harmonize data
dat <- harmonise_data(
    exposure_dat = format_data(exposure_dat, type="exposure"),
    outcome_dat = format_data(outcome_dat, type="outcome")
)

# Run MR analysis
mr_results <- mr(dat, method_list = c("{method}"))

# Capture results
kosmos_capture("mr_results", as.data.frame(mr_results))
kosmos_capture("n_snps", nrow(dat))
kosmos_capture("method", "{method}")

# Print summary
print(mr_results)
'''

        return self.execute(code, capture_results=True)


def is_r_code(code: str) -> bool:
    """
    Quick check if code appears to be R code.

    Args:
        code: Code string to check

    Returns:
        True if code appears to be R
    """
    executor = RExecutor()
    return executor.detect_language(code) == 'r'
