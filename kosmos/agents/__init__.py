"""
Agents module for Kosmos.

Provides specialized agents for different research tasks:
- DataAnalystAgent: Executes data analysis tasks
- LiteratureAnalyzer: Searches and synthesizes literature
- HypothesisGenerator: Generates testable hypotheses
- ExperimentDesigner: Designs experiments
- ResearchDirector: Strategic planning

Gap 3 Enhancement:
- SkillLoader: Loads domain-specific scientific skills for agent prompts
"""

from .skill_loader import SkillLoader
from .data_analyst import DataAnalystAgent
from .hypothesis_generator import HypothesisGeneratorAgent
from .experiment_designer import ExperimentDesignerAgent
from .literature_analyzer import LiteratureAnalyzerAgent
from .research_director import ResearchDirectorAgent

__all__ = [
    "SkillLoader",
    "DataAnalystAgent",
    "HypothesisGeneratorAgent",
    "ExperimentDesignerAgent",
    "LiteratureAnalyzerAgent",
    "ResearchDirectorAgent",
]
