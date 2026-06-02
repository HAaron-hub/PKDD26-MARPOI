# MAPOI (Multi-Agent POI Recommendation)

MAPOI is a multi-agent Point-of-Interest (POI) recommendation system that combines:
- **Habitual analysis** (historical behavior patterns)
- **Temporal analysis** (time/day preferences)
- **Contextual analysis** (weather and date context)
- **Memory and profiling** (global knowledge + user profile updates)

## Project Structure

- `main.py`: Entry point for running recommendation experiments.
- `models/MARPOI.py`: Main MAPOI orchestration pipeline.
- `agents/`: Agent implementations and tool registry.
  - `habitual_analyst.py`
  - `temporal_analyst.py`
  - `contextual_analyst.py`
  - `memory_master.py`
  - `agent_tools.py`
- `utils.py`: Shared utilities (LLM calls, parsers, date helpers, cache).
- `data/`: Input datasets (`nyc`, `tky`, `ca`).
- `memory/`: Generated memory artifacts (`global_memory.json`, `transitions.json`, profiles).
- `output/`, `output_pkdd/`, `results/`: Experiment outputs.

## Requirements

## Conda Environment (`LLMMove2`)

This repository includes an exported Conda environment file:
- `environment.LLMMove2.yml`

Create the environment from YAML:

```bash
conda env create -f environment.LLMMove2.yml
```

Activate the environment:

```bash
conda activate LLMMove2
```

Update an existing environment from YAML:

```bash
conda env update -n LLMMove2 -f environment.LLMMove2.yml --prune
```

Typical Python dependencies used in this project:
- `openai`
- `tenacity`
- `pandas`
- `numpy`
- `tqdm`
- `holidays`
- `meteostat`

Install with:

```bash
pip install openai tenacity pandas numpy tqdm holidays meteostat
```

## Data Files

The current pipeline uses:
- `data/<dataset>/new_train_sample.csv`
- `data/<dataset>/new_test_sample.csv`

Supported datasets:
- `nyc`
- `tky`
- `ca`

## Run

Example:

```bash
python main.py -m MARPOI -d nyc
```

Other dataset options:

```bash
python main.py -m MARPOI -d tky
python main.py -m MARPOI -d ca
```

## Notes

- Weather lookup in `WeatherSearch` uses **hourly data first** and falls back to **daily data** when hourly data is unavailable.
- Memory artifacts are saved under `memory/<dataset>/`.
- Model outputs and intermediate agent outputs are written to `output/` and `output_pkdd/`.

## License

No license file is currently included in this repository. Add one before publishing if needed.
