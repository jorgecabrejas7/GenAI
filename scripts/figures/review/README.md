# Review-presentation figures

Standalone material for the project-review deck. Not part of the eval pipeline.

| file | what | run |
|---|---|---|
| `fig1_pipeline.svg` | Pipeline diagram, flat vector, one accent colour. Edit text in any editor. | open directly |
| `fig2_porosity_control.py` | Cross-sections at rising target porosity + requested-vs-measured scatter (R², MAE). | `python fig2_porosity_control.py --csv por.csv --slices slices/ --slice-porosity 0.005,0.01,...` |
| `fig3_sequence_shape.py` | Grid: stacking sequences (top), shapes/thicknesses (bottom). | `python fig3_sequence_shape.py --images panels/ --manifest panels/manifest.csv` |
| `fig4_real_vs_synthetic.py` | Real vs synthetic slices (`--blind` shuffles + writes a key) and pore-size / grey histograms. | `python fig4_real_vs_synthetic.py --real r/ --synth s/ --pores-real pr.csv --pores-synth ps.csv` |

Each script runs with no arguments on placeholder data and writes `<out>.png` + `<out>.pdf` at 300 dpi.
The style block at the top of each script is identical; edit it in all three to keep the deck consistent.
Data formats are in each script's docstring (`python figN_… -h`).
