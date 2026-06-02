# AgenticRAGCrewAI

AgenticRAGCrewAI is a CrewAI-based assistant for analysing Lake Pyhäjärvi fishery data, economic values, and water-quality relationships using the local `analytics/` package and the processed CSV files in `data/processed/`.

## 1. Project layout

```text
AgenticRAGCrewAI/
├── __init__.py
├── agents.py
├── tools.py
├── tasks.py
├── process.py
├── streamlit_app.py
├── requirements.txt
├── analytics/
│   ├── __init__.py
│   ├── comparisons.py
│   ├── data_loader.py
│   ├── trends.py
│   ├── economic_rankings.py
│   ├── valuations.py
│   ├── relationships.py
│   ├── species_relationships.py
│   ├── series_engine.py
│   └── count_valuations.py
└── data/
    └── processed/
        ├── catch_clean.csv
        ├── count_catch_clean.csv
        ├── luke_clean.csv
        ├── water_quality_clean.csv
        ├── water_quality_long.csv
        ├── species_dictionary.csv
        └── metric_dictionary.csv
```

## 2. Create a virtual environment

Run these commands from the folder that contains `AgenticRAGCrewAI/`.

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Python 3.11 or 3.12 is recommended.

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r AgenticRAGCrewAI/requirements.txt
```

## 4. Configure your API key

Create a `.env` file beside the `AgenticRAGCrewAI/` folder:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 5. Test the analytics layer

Run this before running CrewAI. It verifies that the data files and analytics imports work.

### macOS/Linux

```bash
python - <<'PY'
from AgenticRAGCrewAI.analytics.trends import list_available_species, get_total_catch_by_year
from AgenticRAGCrewAI.analytics.economic_rankings import get_top_value_species_by_year
from AgenticRAGCrewAI.analytics.relationships import correlate_species_with_water_metric

print(list_available_species()[:5])
print(get_total_catch_by_year()[:2])
print(get_top_value_species_by_year(2024, 3))
print(correlate_species_with_water_metric("perch", "total_p_surface_ug_l", 0)["interpretation"])
PY
```

### Windows PowerShell

```powershell
python -c "from AgenticRAGCrewAI.analytics.trends import list_available_species, get_total_catch_by_year; from AgenticRAGCrewAI.analytics.economic_rankings import get_top_value_species_by_year; from AgenticRAGCrewAI.analytics.relationships import correlate_species_with_water_metric; print(list_available_species()[:5]); print(get_total_catch_by_year()[:2]); print(get_top_value_species_by_year(2024, 3)); print(correlate_species_with_water_metric('perch', 'total_p_surface_ug_l', 0)['interpretation'])"
```

## 6. Run the command-line CrewAI workflow

Run from the folder that contains `AgenticRAGCrewAI/`:

```bash
python -m AgenticRAGCrewAI.process
```

Or ask your own question:

```bash
python -m AgenticRAGCrewAI.process "Is crayfish more valuable than fish? When did it become the most important species?"
```

## 7. Run the Streamlit app

Run from the folder that contains `AgenticRAGCrewAI/`:

```bash
streamlit run AgenticRAGCrewAI/streamlit_app.py
```

Then open the local URL shown by Streamlit, usually `http://localhost:8501`.

## 8. What was fixed

- Added `AgenticRAGCrewAI/__init__.py`.
- Added `analytics/__init__.py`.
- Added missing `analytics/comparisons.py` required by `analytics/relationships.py`.
- Converted internal analytics imports to package-relative imports.
- Converted `tools.py` imports to package-relative imports.
- Added typed `name: str` and `description: str` fields for CrewAI/Pydantic compatibility.
- Added missing `crewai-tools` to `requirements.txt`.
- Removed unsupported `max_steps` argument from `crew.kickoff(...)`.
- Replaced JSON dumping of CrewAI result objects with safe text conversion.
- Fixed Streamlit imports and display logic.
- Disabled `human_input=True` pauses so first runs do not appear stuck.


## Troubleshooting: BaseTool import

This project uses the current CrewAI custom tool import:

```python
from crewai.tools import BaseTool
```

Do not change it back to `from crewai_tools import BaseTool`. Newer `crewai-tools` versions may not export `BaseTool`, which causes:

```text
ImportError: cannot import name 'BaseTool' from 'crewai_tools'
```

If you already installed the old dependency set, reinstall after this update:

```bash
pip uninstall -y crewai-tools
pip install -r AgenticRAGCrewAI/requirements.txt
```
