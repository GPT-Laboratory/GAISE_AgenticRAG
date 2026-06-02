"""
Task definitions for the AgenticRAGCrewAI fishery assistant.

Tasks define the discrete units of work that agents execute. The task
sequence is intentionally broad so the demo can show catch trends,
economic valuation, environmental correlations, and final synthesis.
"""

from __future__ import annotations

from crewai import Agent, Task

from agents import create_agents


def create_tasks(question: str, agents: dict[str, Agent] | None = None) -> list[Task]:
    """Create the CrewAI task list for a user question.

    Args:
        question: The user question driving the analysis.
        agents: Optional existing agent instances. Passing these from
            process.py keeps task agents identical to the crew agents.

    Returns:
        A list of CrewAI Task instances representing the workflow.
    """
    if agents is None:
        agents = create_agents()

    return [
        Task(
            description=(
                f"User question: {question}\n\n"
                "List all fish and crayfish species present in the catch dataset. "
                "Return them as a concise bullet list."
            ),
            expected_output="A concise bullet list of species keys/names available in the dataset.",
            agent=agents["data_specialist"],
            human_input=False,
        ),
        Task(
            description=(
                f"User question: {question}\n\n"
                "Compute the total catch in kilograms for each available year. "
                "Return year and total catch values, and highlight the overall trend."
            ),
            expected_output="A list/table of year and total_catch_kg values plus a short trend summary.",
            agent=agents["data_specialist"],
            human_input=False,
        ),
        Task(
            description=(
                f"User question: {question}\n\n"
                "Identify which species experienced the largest percentage increase in catch "
                "over the available time period."
            ),
            expected_output="A dictionary-style summary with species_key, start_year, end_year, percent_change, and direction.",
            agent=agents["data_specialist"],
            human_input=False,
        ),
        Task(
            description=(
                f"User question: {question}\n\n"
                "Estimate the economic value of all species for 2024 and list the top three species by value. "
                "Use the available catch and price data only."
            ),
            expected_output="The top three species for 2024 with estimated values in euros and supporting values used.",
            agent=agents["economic_analyst"],
            human_input=False,
        ),
        Task(
            description=(
                f"User question: {question}\n\n"
                "Compute the same-year correlation between perch catch in kilograms and surface phosphorus "
                "concentration using metric_key='total_p_surface_ug_l' and lag_years=0."
            ),
            expected_output="Correlation coefficient, interpretation, number of years used, and a note that correlation does not imply causation.",
            agent=agents["environment_analyst"],
            human_input=False,
        ),
        Task(
            description=(
                f"User question: {question}\n\n"
                "Synthesize the previous task outputs into a clear final answer. "
                "Prioritize directly answering the user question, cite specific years/species/values when available, "
                "and clearly state any limitations of the data."
            ),
            expected_output="A clear, well-structured final report answering the user question with evidence from the previous analyses.",
            agent=agents["reporter"],
            human_input=False,
        ),
    ]
