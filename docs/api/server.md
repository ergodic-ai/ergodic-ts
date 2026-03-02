# Dashboard Server

FastAPI backend for the backtest dashboard, serving both a REST API and a
static frontend.

## Quick start

```bash
# Launch the dashboard
uv run --extra forecast python -m ergodicts.server --runs-dir runs --port 8765
```

Then open `http://localhost:8765` in your browser.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/runs` | List all saved and in-progress runs |
| `GET` | `/api/runs/{run_id}` | Metadata for a single run |
| `GET` | `/api/runs/{run_id}/summary` | Per-node mean metrics table |
| `GET` | `/api/runs/{run_id}/charts` | Plotly JSON figures per node |
| `GET` | `/api/runs/{run_id}/fold/{fold_idx}` | Detailed fold data (forecasts, actuals, metrics) |
| `GET` | `/api/runs/{run_id}/fold/{fold_idx}/param-cards` | Parameter cards per node |
| `GET` | `/api/runs/{run_id}/fold/{fold_idx}/decomposition` | Per-component decomposition |
| `GET` | `/api/compare?run_a=...&run_b=...` | Side-by-side run comparison |
| `POST` | `/api/runs` | Launch a new backtest run |
| `GET` | `/api/runs/{run_id}/progress` | SSE stream of run progress |
| `GET` | `/api/components` | Component library catalogue |

## Request / response models

::: ergodicts.server.RunListItem

::: ergodicts.server.LaunchRequest

::: ergodicts.server.LaunchResponse

## App factory

::: ergodicts.server.create_app

## CLI entry point

::: ergodicts.server.serve
