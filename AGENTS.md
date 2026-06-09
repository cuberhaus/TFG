# TFG — Polyp Detection (Bachelor's Thesis)

FIB-UPC Bachelor's Thesis: deep learning system for polyp detection in colonoscopy. Trains Faster R-CNN, RetinaNet, and SSD on data augmented by CycleGAN and SPADE generative models, with hyperparameter search via Optuna and Ray Tune.

## Architecture

- **ML core** (`code/src/`): PyTorch + torchvision training, evaluation, and generative augmentation. Shared utilities in `code/src/clases/` (`model_utils.py`, `custom_dataset.py`, `cyclegan.py`).
- **Backend** (`backend/main.py`): FastAPI REST API wrapping the PyTorch inference and training pipelines; subprocess-driven for long-running jobs.
- **Frontend** (`frontend/`): React 19 + Vite + Tailwind CSS SPA, axios for API calls, tab-based dashboard (one component per tab).
- **Legacy UI** (`code/src/app.py`): Streamlit dashboard kept for historical reference.

## Build and Test

```bash
make install     # backend pip + code pip + frontend npm
make run         # backend on :8082 and Vite dev server on :5173 in parallel
make backend     # FastAPI only (cd backend && ./run.sh)
make frontend    # Vite only (cd frontend && npm run dev)

cd frontend && npm run build && npm run lint
cd backend && pytest test_main.py
```

CLI training entry points live under `code/src/` (see [README.md](README.md) for the full list).

## Conventions

- **Backend**: FastAPI with Pydantic models and dependency injection; run blocking ML inference in thread pools, not the event loop. Wrap existing PyTorch code in `clases.model_utils` rather than reimplementing it. Endpoints accept image uploads and return JSON (bounding boxes, scores) or annotated images.
- **Frontend**: React functional components + Hooks, Tailwind for styling, the shared axios instance in `frontend/src/api.ts` (honors `VITE_API_URL`). One component per dashboard tab under `frontend/src/components/`.
- **Python**: 3.11+; dependencies in `backend/requirements.txt` and `code/requirements.txt` (kept separate on purpose).

## Agent skills

Installable skills live under `.agents/skills/` (gitignored; restore with `make skills-restore`). Pinned versions are in [skills-lock.json](skills-lock.json).

- **fastapi-templates** — consult when adding or modifying endpoints in [backend/](backend/).
- **pytorch-patterns** — consult when editing model/training/inference code under `code/src/` (including `clases/model_utils.py`, `cyclegan.py`).
- **vercel-react-best-practices** — consult when editing React 19 components in [frontend/](frontend/).
- **pytest-coverage** — consult when adding or improving backend tests (`backend/test_main.py`).

## Pitfalls

- **DO NOT delete or modify `code/src/app.py`** — the legacy Streamlit dashboard is preserved as a historical reference and legacy test environment.
- `docker-compose.mlops.yml` runs under its own Compose project (`tfg-mlops`) so external scripts that `docker compose down --remove-orphans` on the main `tfg` project do not tear it down. Keep the project names separate.
- Dataset (`code/data/`) and outputs (`code/out/`, `out/`) are git-ignored; the LDPolypVideo layout in [README.md](README.md) must be reproduced manually before training.

See [README.md](README.md) for thesis context, dataset setup, and full CLI reference.
