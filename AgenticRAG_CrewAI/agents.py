"""
Agent definitions for the Bio AI fishery assistant.

This module defines a set of CrewAI agents with distinct roles and
capabilities. Each agent is configured with a goal, backstory and
assigned tools that encapsulate domain-specific logic. Autonomy is
controlled via the ``max_iterations`` and ``allow_delegation``
parameters.

Agents are intentionally scoped: the ``DataSpecialist`` focuses on
descriptive statistics and trends, the ``EconomicAnalyst`` handles
valuation and economic comparisons, the ``EnvironmentAnalyst`` deals
with environmental correlations, and the ``Reporter`` synthesises
results into a final answer. This separation of concerns allows
participants to reason about how different roles collaborate and how
autonomy can be tuned per agent.
"""

from __future__ import annotations

from crewai import Agent

from tools import (
    ListSpeciesTool,
    TotalCatchTrendTool,
    TotalCatchForYearTool,
    SpeciesTrendTool,
    TopSpeciesByYearTool,
    LargestSpeciesChangeTool,
    EstimateSpeciesValueTool,
    TopValueSpeciesByYearTool,
    TopValueSpeciesOverTimeTool,
    ValueCrossoverYearTool,
    SpeciesValueTrendTool,
    CompareSpeciesValuesTool,
    SpeciesMetricCorrelationTool,
    CombinedSpeciesMetricCorrelationTool,
)


def create_agents() -> dict[str, Agent]:
    """Instantiate and return all agents used in the Bio AI Crew."""
    data_specialist = Agent(
        role="Fishery Data Specialist",
        goal=(
            "Provide accurate and complete information about fish catch trends and species statistics "
            "for Lake Pyhäjärvi."
        ),
        backstory=(
            "You work for the Finnish Environment Institute and maintain the official fish catch records "
            "for Lake Pyhäjärvi. You are meticulous, provide numeric results with context, and never "
            "hallucinate."
        ),
        tools=[
            ListSpeciesTool(),
            TotalCatchTrendTool(),
            TotalCatchForYearTool(),
            SpeciesTrendTool(),
            TopSpeciesByYearTool(),
            LargestSpeciesChangeTool(),
        ],
        memory=True,
        max_iterations=3,
        allow_delegation=False,
    )

    economic_analyst = Agent(
        role="Economic Analyst",
        goal=(
            "Analyse fishery catch data and market prices to estimate economic value and compare species."
        ),
        backstory=(
            "A seasoned economist who specialises in fisheries. You use market data and catch volumes "
            "to compute valuations, rank species by value and determine when one species becomes more "
            "valuable than another."
        ),
        tools=[
            EstimateSpeciesValueTool(),
            TopValueSpeciesByYearTool(),
            TopValueSpeciesOverTimeTool(),
            ValueCrossoverYearTool(),
            SpeciesValueTrendTool(),
            CompareSpeciesValuesTool(),
        ],
        memory=True,
        max_iterations=4,
        allow_delegation=True,
    )

    environment_analyst = Agent(
        role="Environmental Analyst",
        goal=(
            "Investigate relationships between fish catch and water quality metrics such as nutrient concentrations and temperature."
        ),
        backstory=(
            "You are a limnologist studying the interplay between aquatic ecosystems and fisheries. You look for correlations and patterns "
            "between environmental variables and fish populations."
        ),
        tools=[
            SpeciesMetricCorrelationTool(),
            CombinedSpeciesMetricCorrelationTool(),
        ],
        memory=True,
        max_iterations=4,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Science Communicator",
        goal=(
            "Summarise analytical results from the other agents into a clear, concise and well-structured narrative for the user."
        ),
        backstory=(
            "You are skilled at translating complex quantitative findings into accessible prose. You review the outputs from other agents, "
            "check for consistency and produce a final answer that addresses the user’s question."
        ),
        tools=[],
        memory=True,
        max_iterations=2,
        allow_delegation=False,
    )

    return {
        "data_specialist": data_specialist,
        "economic_analyst": economic_analyst,
        "environment_analyst": environment_analyst,
        "reporter": reporter,
    }
