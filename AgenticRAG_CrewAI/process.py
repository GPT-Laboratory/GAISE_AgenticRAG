"""Command-line entry point for the AgenticRAGCrewAI workflow."""

from __future__ import annotations

import argparse
from typing import Any

from crewai import Crew, Process
from dotenv import load_dotenv

from agents import create_agents
from tasks import create_tasks

load_dotenv()


def run_workflow(question: str, max_steps: int = 30) -> Any:
    """Run the CrewAI workflow for a given natural-language question.

    The ``max_steps`` argument is kept for backward compatibility with
    earlier versions of this project. Current CrewAI versions control
    agent iteration limits through each Agent's ``max_iterations`` setting,
    so ``max_steps`` is not passed to ``crew.kickoff``.
    """
    agents = create_agents()
    tasks = create_tasks(question, agents=agents)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff(inputs={"question": question})


def result_to_text(result: Any) -> str:
    """Convert a CrewAI result object into displayable text."""
    raw = getattr(result, "raw", None)
    if raw:
        return str(raw)
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgenticRAGCrewAI assistant")
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default="Is crayfish more valuable than fish? When did it become the most important species?",
        help="Question to ask the assistant",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Kept for compatibility; agent max_iterations are used by CrewAI.",
    )
    args = parser.parse_args()

    final_result = run_workflow(args.question, args.max_steps)
    print("\nFINAL RESULT:\n")
    print(result_to_text(final_result))


if __name__ == "__main__":
    main()
