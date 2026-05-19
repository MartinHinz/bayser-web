# Bayser Web

Bayser Web is a small Streamlit frontend for [Bayser](https://github.com/MartinHinz/bayser), a Python package for radiocarbon-informed Bayesian seriation.

The web app is intended for quick exploratory runs. Full analyses, longer MCMC runs, model comparison, and publication-grade workflows should be carried out locally with the Bayser command-line interface.

## Online workflow

1. Upload a binary assemblage-by-type feature matrix as CSV.
2. Optionally upload a radiocarbon table.
3. Check the column mapping and input preview.
4. Run the limited online model.
5. Inspect posterior ranks, diagnostic plots, outlier suggestions, and download the complete run.

## Online limitations

The public online version uses a limited quick preset:

* 500 tuning steps per chain
* 500 posterior draws per chain
* 2 chains
* maximum runtime: 2 minutes

Standard and careful runs should be executed locally.

## Input format

### Feature matrix

The feature matrix should contain assemblages in rows and artefact types in columns.

If no explicit feature ID column is supplied, the first column is used as the assemblage ID.

Example:

```csv
grave_id,type_a,type_b,type_c
G1,1,0,1
G2,0,1,1
G3,1,1,0
```

Values are converted to binary presence/absence by Bayser.

### Radiocarbon table

Radiocarbon-linked runs require a table with:

* an assemblage ID column
* a radiocarbon age column
* a radiocarbon error column

Example:

```csv
grave_id,bp,error
G1,2450,30
G2,2380,25
```

The radiocarbon IDs must match the assemblage IDs in the feature matrix.

## Calibration curve

The app bundles an IntCal20 calibration curve at:

```text
bayser_app/assets/intcal20.14c
```

Users may upload a custom calibration curve to override the bundled file.

Please cite IntCal20 when using radiocarbon-linked runs:

Reimer, P. J. et al. 2020. The IntCal20 Northern Hemisphere Radiocarbon Age Calibration Curve (0–55 cal kBP). *Radiocarbon* 62(4): 725–757. [https://doi.org/10.1017/RDC.2020.41](https://doi.org/10.1017/RDC.2020.41)

## Local development

```bash
uv sync
uv run streamlit run app.py
```

The app calls Bayser in a subprocess using the current Python environment.

## Related repository

Core package:

[https://github.com/MartinHinz/bayser](https://github.com/MartinHinz/bayser)

## License

Bayser Web is released under the MIT License. See `LICENSE` for details.

The bundled IntCal20 calibration curve is not part of the software license. Please cite Reimer et al. (2020) when using radiocarbon-linked runs.