# Village data directory

Place downloaded village bundles (from the project's **Get started** page) inside this `data/` directory. Each bundle should have the structure:

```
data/<village_slug>/
  input.geojson
  imagery.tif
  boundaries.tif
  example_truths.geojson
```

From the repository root you can run the worked example with:

```
uv run quickstart.py data/<village_slug>
```
