# Legacy one-shot experiments

These files are retained from the earlier terminal-driven prototype for reference only. They are not imported by the Docker runtime. The active application starts through Docker Compose and uses `src.training.worker`, `src.api.main`, and `gui/`.

The original `training.txt` and `test.txt` files are in `data/`. The scripts were designed around that manually supplied dataset and should not be used to evaluate the continuous trainer's Wikipedia corpus.
