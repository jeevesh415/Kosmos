"""
Convergence Detection - Detects when autonomous research should stop (Phase 7).

Implements mandatory and optional stopping criteria:
- Mandatory: Iteration limit, no testable hypotheses
- Optional: Novelty decline, diminishing returns
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from pydantic import BaseModel, Field
from enum import Enum
import logging
import numpy as np

from kosmos.core.workflow import ResearchPlan
from kosmos.models.hypothesis import Hypothesis
from kosmos.models.result import ExperimentResult
from kosmos.utils.compat import model_to_dict

logger = logging.getLogger(__name__)


class StoppingReason(str, Enum):
    """Reasons for stopping research."""

    ITERATION_LIMIT = "iteration_limit"
    NO_TESTABLE_HYPOTHESES = "no_testable_hypotheses"
    NOVELTY_DECLINE = "novelty_decline"
    DIMINISHING_RETURNS = "diminishing_returns"
    ALL_HYPOTHESES_TESTED = "all_hypotheses_tested"
    USER_REQUESTED = "user_requested"


class ConvergenceMetrics(BaseModel):
    """Metrics for convergence detection."""

    # Discovery metrics
    discovery_rate: float = Field(0.0, description="Significant results per experiment")
    total_experiments: int = 0
    significant_results: int = 0

    # Novelty metrics
    novelty_score: float = Field(0.0, description="Current novelty score")
    novelty_trend: List[float] = Field(default_factory=list, description="Novelty over time")
    novelty_declining: bool = False

    # Saturation metrics
    saturation_ratio: float = Field(0.0, description="Tested / total hypotheses")
    hypotheses_tested: int = 0
    total_hypotheses: int = 0

    # Consistency metrics
    consistency_score: float = Field(0.0, description="Replication rate")
    replicated_findings: int = 0
    total_findings: int = 0

    # Iteration tracking
    iteration_count: int = 0
    max_iterations: int = 10

    # Cost metrics
    total_cost: float = 0.0
    cost_per_discovery: Optional[float] = None

    # Timestamps
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update_timestamp(self):
        """Update last_update timestamp."""
        self.last_update = datetime.now(timezone.utc)


class StoppingDecision(BaseModel):
    """Decision on whether to stop research."""

    should_stop: bool
    reason: StoppingReason
    is_mandatory: bool  # True if mandatory criterion, False if optional
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in decision")
    details: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConvergenceReport(BaseModel):
    """Comprehensive convergence report."""

    research_question: str
    converged: bool
    stopping_reason: Optional[StoppingReason] = None

    # Summary statistics
    total_iterations: int
    total_hypotheses: int
    hypotheses_supported: int
    hypotheses_rejected: int
    experiments_conducted: int

    # Key findings
    key_discoveries: List[Dict[str, Any]] = Field(default_factory=list)
    supported_hypotheses: List[str] = Field(default_factory=list)
    rejected_hypotheses: List[str] = Field(default_factory=list)

    # Metrics
    final_metrics: ConvergenceMetrics

    # Recommendations
    research_complete: bool
    recommended_next_steps: List[str] = Field(default_factory=list)

    # Report content
    summary: str = ""
    detailed_report: str = ""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        """Export report as markdown."""
        md = f"""# Convergence Report

**Research Question**: {self.research_question}

**Status**: {"✅ Research Converged" if self.converged else "🔄 Research Ongoing"}

**Stopping Reason**: {self.stopping_reason.value if self.stopping_reason else "N/A"}

---

## Summary Statistics

- **Total Iterations**: {self.total_iterations}
- **Hypotheses Generated**: {self.total_hypotheses}
- **Hypotheses Supported**: {self.hypotheses_supported}
- **Hypotheses Rejected**: {self.hypotheses_rejected}
- **Experiments Conducted**: {self.experiments_conducted}

---

## Key Metrics

- **Discovery Rate**: {self.final_metrics.discovery_rate:.2%}
- **Novelty Score**: {self.final_metrics.novelty_score:.2f}
- **Saturation**: {self.final_metrics.saturation_ratio:.2%}
- **Consistency**: {self.final_metrics.consistency_score:.2%}

---

## Supported Hypotheses

{chr(10).join(f"- {h}" for h in self.supported_hypotheses) if self.supported_hypotheses else "None"}

---

## Recommended Next Steps

{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(self.recommended_next_steps))}

---

## Detailed Summary

{self.summary}

---

**Generated**: {self.generated_at.isoformat()}
"""
        return md


class ConvergenceDetector:
    """
    Detects when autonomous research should converge/stop.

    Monitors progress metrics and applies stopping criteria.
    """

    def __init__(
        self,
        mandatory_criteria: Optional[List[str]] = None,
        optional_criteria: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize convergence detector.

        Args:
            mandatory_criteria: List of mandatory stopping criteria
            optional_criteria: List of optional stopping criteria
            config: Configuration dict
        """
        self.mandatory_criteria = mandatory_criteria or ["iteration_limit", "no_testable_hypotheses"]
        self.optional_criteria = optional_criteria or ["novelty_decline", "diminishing_returns"]

        # Configuration
        self.config = config or {}
        self.novelty_decline_threshold = self.config.get("novelty_decline_threshold", 0.3)
        max_iters = self.config.get("max_iterations", 10)
        self.novelty_decline_window = self.config.get(
            "novelty_decline_window",
            min(5, max(2, max_iters - 1))
        )
        self.cost_per_discovery_threshold = self.config.get("cost_per_discovery_threshold", 1000.0)
        self.min_experiments_before_convergence = self.config.get(
            "min_experiments_before_convergence", 2
        )

        # Metrics tracking
        self.metrics = ConvergenceMetrics()

        logger.info(
            f"ConvergenceDetector initialized (mandatory: {self.mandatory_criteria}, "
            f"optional: {self.optional_criteria})"
        )

    # ========================================================================
    # MAIN CONVERGENCE CHECK
    # ========================================================================

    def check_convergence(
        self,
        research_plan: ResearchPlan,
        hypotheses: List[Hypothesis],
        results: List[ExperimentResult],
        total_cost: Optional[float] = None
    ) -> StoppingDecision:
        """
        Check if research should stop.

        Args:
            research_plan: Current research plan
            hypotheses: All hypotheses
            results: All experiment results
            total_cost: Optional accumulated cost from LLM provider

        Returns:
            StoppingDecision: Decision on whether to stop
        """
        logger.info("Checking convergence criteria")

        # Update metrics
        self._update_metrics(research_plan, hypotheses, results, total_cost=total_cost)

        # Check hard-stop mandatory criteria first (e.g., no_testable_hypotheses)
        for criterion in self.mandatory_criteria:
            if criterion == "iteration_limit":
                continue  # Check after optional criteria
            decision = self._check_criterion(criterion, research_plan, hypotheses, results, mandatory=True)
            if decision.should_stop:
                logger.info(f"Mandatory stopping criterion met: {criterion}")
                return decision

        # Check optional (scientific) criteria before iteration_limit
        # so scientific convergence reasons are reported when both would fire
        for criterion in self.optional_criteria:
            decision = self._check_criterion(criterion, research_plan, hypotheses, results, mandatory=False)
            if decision.should_stop:
                logger.info(f"Optional stopping criterion met: {criterion}")
                return decision

        # Check iteration_limit last among mandatory criteria
        if "iteration_limit" in self.mandatory_criteria:
            decision = self._check_criterion("iteration_limit", research_plan, hypotheses, results, mandatory=True)
            if decision.should_stop:
                logger.info("Mandatory stopping criterion met: iteration_limit")
                return decision

        # No stopping criteria met
        return StoppingDecision(
            should_stop=False,
            reason=StoppingReason.USER_REQUESTED,  # Placeholder
            is_mandatory=False,
            confidence=1.0,
            details="No stopping criteria met, research continues"
        )

    def _check_criterion(
        self,
        criterion: str,
        research_plan: ResearchPlan,
        hypotheses: List[Hypothesis],
        results: List[ExperimentResult],
        mandatory: bool
    ) -> StoppingDecision:
        """Check a specific stopping criterion."""

        if criterion == "iteration_limit":
            return self.check_iteration_limit(research_plan)

        elif criterion == "no_testable_hypotheses":
            return self.check_hypothesis_exhaustion(research_plan, hypotheses)

        elif criterion == "novelty_decline":
            return self.check_novelty_decline()

        elif criterion == "diminishing_returns":
            return self.check_diminishing_returns()

        else:
            logger.warning(f"Unknown criterion: {criterion}")
            return StoppingDecision(
                should_stop=False,
                reason=StoppingReason.USER_REQUESTED,
                is_mandatory=mandatory,
                confidence=0.0,
                details=f"Unknown criterion: {criterion}"
            )

    # ========================================================================
    # MANDATORY STOPPING CRITERIA
    # ========================================================================

    def check_iteration_limit(self, research_plan: ResearchPlan) -> StoppingDecision:
        """
        Check if iteration limit reached.

        Defers convergence when fewer than min_experiments_before_convergence
        experiments have completed AND testable work remains (untested hypotheses
        or queued experiments). This prevents premature stopping when the pipeline
        generated hypotheses but only tested a small fraction.

        Hard caps (MAX_ACTIONS_PER_ITERATION, max_runtime_hours) still apply
        independently, so this cannot cause infinite loops.

        Args:
            research_plan: Research plan

        Returns:
            StoppingDecision
        """
        at_limit = research_plan.iteration_count >= research_plan.max_iterations

        if at_limit:
            completed = len(research_plan.completed_experiments)
            untested = research_plan.get_untested_hypotheses()
            has_testable_work = len(untested) > 0 or len(research_plan.experiment_queue) > 0

            if completed < self.min_experiments_before_convergence and has_testable_work:
                logger.info(
                    f"Iteration limit reached but deferring convergence: "
                    f"{completed}/{self.min_experiments_before_convergence} experiments completed, "
                    f"testable work remains ({len(untested)} untested, "
                    f"{len(research_plan.experiment_queue)} queued)"
                )
                return StoppingDecision(
                    should_stop=False,
                    reason=StoppingReason.ITERATION_LIMIT,
                    is_mandatory=True,
                    confidence=0.5,
                    details=(
                        f"Iteration {research_plan.iteration_count}/{research_plan.max_iterations} "
                        f"but only {completed}/{self.min_experiments_before_convergence} experiments "
                        f"completed — deferring"
                    )
                )

        return StoppingDecision(
            should_stop=at_limit,
            reason=StoppingReason.ITERATION_LIMIT,
            is_mandatory=True,
            confidence=1.0,
            details=f"Iteration {research_plan.iteration_count}/{research_plan.max_iterations}"
        )

    def check_hypothesis_exhaustion(
        self,
        research_plan: ResearchPlan,
        hypotheses: List[Hypothesis]
    ) -> StoppingDecision:
        """
        Check if no testable hypotheses remain.

        Args:
            research_plan: Research plan
            hypotheses: All hypotheses

        Returns:
            StoppingDecision
        """
        untested = research_plan.get_untested_hypotheses()
        no_experiments_queued = len(research_plan.experiment_queue) == 0

        should_stop = len(untested) == 0 and no_experiments_queued

        return StoppingDecision(
            should_stop=should_stop,
            reason=StoppingReason.NO_TESTABLE_HYPOTHESES,
            is_mandatory=True,
            confidence=1.0,
            details=f"{len(untested)} untested hypotheses, {len(research_plan.experiment_queue)} queued experiments"
        )

    # ========================================================================
    # OPTIONAL STOPPING CRITERIA
    # ========================================================================

    def check_novelty_decline(self) -> StoppingDecision:
        """
        Check if novelty is declining.

        Returns:
            StoppingDecision
        """
        trend = self.metrics.novelty_trend

        if len(trend) < self.novelty_decline_window:
            # Not enough data
            return StoppingDecision(
                should_stop=False,
                reason=StoppingReason.NOVELTY_DECLINE,
                is_mandatory=False,
                confidence=0.0,
                details=f"Insufficient data ({len(trend)}/{self.novelty_decline_window} points)"
            )

        # Check last N values
        recent = trend[-self.novelty_decline_window:]

        # All below threshold?
        all_below = all(v < self.novelty_decline_threshold for v in recent)

        # Declining trend?
        is_declining = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))

        should_stop = all_below or is_declining

        return StoppingDecision(
            should_stop=should_stop,
            reason=StoppingReason.NOVELTY_DECLINE,
            is_mandatory=False,
            confidence=0.8 if should_stop else 0.2,
            details=f"Recent novelty: {recent}, threshold: {self.novelty_decline_threshold}"
        )

    def check_diminishing_returns(self) -> StoppingDecision:
        """
        Check if cost per discovery is too high.

        Returns:
            StoppingDecision
        """
        if self.metrics.cost_per_discovery is None:
            return StoppingDecision(
                should_stop=False,
                reason=StoppingReason.DIMINISHING_RETURNS,
                is_mandatory=False,
                confidence=0.0,
                details="No cost data available"
            )

        should_stop = self.metrics.cost_per_discovery > self.cost_per_discovery_threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason=StoppingReason.DIMINISHING_RETURNS,
            is_mandatory=False,
            confidence=0.7 if should_stop else 0.3,
            details=f"Cost per discovery: ${self.metrics.cost_per_discovery:.2f} (threshold: ${self.cost_per_discovery_threshold:.2f})"
        )

    # ========================================================================
    # PROGRESS METRICS
    # ========================================================================

    def calculate_discovery_rate(self, results: List[ExperimentResult]) -> float:
        """
        Calculate discovery rate: significant results / total experiments.

        Args:
            results: All experiment results

        Returns:
            float: Discovery rate (0.0 - 1.0)
        """
        if not results:
            return 0.0

        significant = sum(1 for r in results if r.supports_hypothesis is True)

        return significant / len(results)

    def calculate_novelty_decline(self, hypotheses: List[Hypothesis]) -> Tuple[float, bool]:
        """
        Calculate current novelty and detect declining trend.

        Args:
            hypotheses: All hypotheses (ordered by creation time)

        Returns:
            Tuple of (current_novelty, is_declining)
        """
        if not hypotheses:
            return 0.0, False

        # Get novelty scores
        novelty_scores = [h.novelty_score for h in hypotheses if h.novelty_score is not None]

        if not novelty_scores:
            return 0.0, False

        current_novelty = novelty_scores[-1] if novelty_scores else 0.0

        # Check trend
        if len(novelty_scores) >= 3:
            # Simple trend: are recent scores declining?
            recent = novelty_scores[-3:]
            is_declining = all(recent[i] >= recent[i+1] for i in range(len(recent)-1))
        else:
            is_declining = False

        return current_novelty, is_declining

    def calculate_saturation(self, research_plan: ResearchPlan) -> float:
        """
        Calculate saturation: tested / total hypotheses.

        Args:
            research_plan: Research plan

        Returns:
            float: Saturation ratio (0.0 - 1.0)
        """
        return research_plan.get_testability_rate()

    def calculate_consistency(self, results: List[ExperimentResult]) -> float:
        """
        Calculate consistency: replication rate.

        For now, simplified as support rate.

        Args:
            results: All experiment results

        Returns:
            float: Consistency score (0.0 - 1.0)
        """
        if not results:
            return 0.0

        supported = sum(1 for r in results if r.supports_hypothesis is True)

        return supported / len(results)

    def _update_metrics(
        self,
        research_plan: ResearchPlan,
        hypotheses: List[Hypothesis],
        results: List[ExperimentResult],
        total_cost: Optional[float] = None
    ):
        """Update all metrics."""

        # Discovery metrics
        self.metrics.discovery_rate = self.calculate_discovery_rate(results)
        self.metrics.total_experiments = len(results)
        self.metrics.significant_results = sum(1 for r in results if r.supports_hypothesis is True)

        # Novelty metrics
        current_novelty, is_declining = self.calculate_novelty_decline(hypotheses)
        self.metrics.novelty_score = current_novelty
        self.metrics.novelty_trend.append(current_novelty)
        self.metrics.novelty_declining = is_declining

        # Saturation metrics
        self.metrics.saturation_ratio = self.calculate_saturation(research_plan)
        self.metrics.hypotheses_tested = len(research_plan.tested_hypotheses)
        self.metrics.total_hypotheses = len(research_plan.hypothesis_pool)

        # Consistency metrics
        self.metrics.consistency_score = self.calculate_consistency(results)

        # Iteration tracking
        self.metrics.iteration_count = research_plan.iteration_count
        self.metrics.max_iterations = research_plan.max_iterations

        # Cost metrics
        if total_cost is not None:
            self.metrics.total_cost = total_cost
        if self.metrics.significant_results > 0 and self.metrics.total_cost > 0:
            self.metrics.cost_per_discovery = self.metrics.total_cost / self.metrics.significant_results
        else:
            self.metrics.cost_per_discovery = None

        self.metrics.update_timestamp()

    # ========================================================================
    # CONVERGENCE REPORT GENERATION
    # ========================================================================

    def generate_convergence_report(
        self,
        research_plan: ResearchPlan,
        hypotheses: List[Hypothesis],
        results: List[ExperimentResult],
        stopping_reason: Optional[StoppingReason] = None
    ) -> ConvergenceReport:
        """
        Generate comprehensive convergence report.

        Args:
            research_plan: Research plan
            hypotheses: All hypotheses
            results: All results
            stopping_reason: Why research stopped

        Returns:
            ConvergenceReport
        """
        logger.info("Generating convergence report")

        # Update metrics one final time
        self._update_metrics(research_plan, hypotheses, results)

        # Extract supported/rejected hypotheses
        supported_hyps = [h.statement for h in hypotheses if h.id in research_plan.supported_hypotheses]
        rejected_hyps = [h.statement for h in hypotheses if h.id in research_plan.rejected_hypotheses]

        # Generate summary
        summary = self._generate_summary(research_plan, hypotheses, results)

        # Recommend next steps
        next_steps = self._recommend_next_steps(research_plan, hypotheses, results)

        report = ConvergenceReport(
            research_question=research_plan.research_question,
            converged=research_plan.has_converged,
            stopping_reason=stopping_reason,
            total_iterations=research_plan.iteration_count,
            total_hypotheses=len(hypotheses),
            hypotheses_supported=len(research_plan.supported_hypotheses),
            hypotheses_rejected=len(research_plan.rejected_hypotheses),
            experiments_conducted=len(results),
            supported_hypotheses=supported_hyps,
            rejected_hypotheses=rejected_hyps,
            final_metrics=self.metrics,
            research_complete=research_plan.has_converged,
            recommended_next_steps=next_steps,
            summary=summary
        )

        logger.info("Convergence report generated")
        return report

    def _generate_summary(
        self,
        research_plan: ResearchPlan,
        hypotheses: List[Hypothesis],
        results: List[ExperimentResult]
    ) -> str:
        """Generate summary text."""

        summary = f"""Research on "{research_plan.research_question}" completed after {research_plan.iteration_count} iterations.

Generated {len(hypotheses)} hypotheses and conducted {len(results)} experiments.

Results:
- {len(research_plan.supported_hypotheses)} hypotheses supported
- {len(research_plan.rejected_hypotheses)} hypotheses rejected
- {len(research_plan.hypothesis_pool) - len(research_plan.tested_hypotheses)} hypotheses remain untested

Discovery rate: {self.metrics.discovery_rate:.1%}
Final novelty score: {self.metrics.novelty_score:.2f}
Research saturation: {self.metrics.saturation_ratio:.1%}
"""

        return summary

    def _recommend_next_steps(
        self,
        research_plan: ResearchPlan,
        hypotheses: List[Hypothesis],
        results: List[ExperimentResult]
    ) -> List[str]:
        """Recommend next steps based on results."""

        recommendations = []

        # If supported hypotheses, recommend replication
        if research_plan.supported_hypotheses:
            recommendations.append("Replicate supported hypotheses in larger studies")

        # If untested hypotheses, recommend continuing
        untested = research_plan.get_untested_hypotheses()
        if untested:
            recommendations.append(f"Test remaining {len(untested)} hypotheses")

        # If high novelty, recommend exploring related areas
        if self.metrics.novelty_score > 0.7:
            recommendations.append("Explore related high-novelty areas")

        # If low discovery rate, recommend refining approach
        if self.metrics.discovery_rate < 0.2:
            recommendations.append("Refine experimental approach to increase discovery rate")

        # Always recommend documenting findings
        recommendations.append("Document findings and prepare publication")

        return recommendations

    # ========================================================================
    # STATUS & REPORTING
    # ========================================================================

    def get_metrics(self) -> ConvergenceMetrics:
        """Get current metrics."""
        return self.metrics

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        return model_to_dict(self.metrics)
