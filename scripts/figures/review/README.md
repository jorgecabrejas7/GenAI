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

## On the campaign data

`extract_review_data.py` builds the inputs from campaigns 18/19/20 and draws all figures into
`runs/campaigns/18-eval-v4-final/figures_review/` (fig1 copied there too). Choices: fig 2 uses the 42
in-range porosity_global cases (0.15 is above the training clamp and excluded) and DDIM-200 seed-101
through-thickness slices; fig 3 uses layups A and C plus three never-seen orders (campaign 20), ply
pitch 8/32 vox, and the L-bracket, taper and letters requests (campaign 19); fig 4 pairs the Na_05
reference crops with the microstructure volumes at phi 0.01/0.03/0.06, pore sizes are connected-
component equivalent diameters over all 18 real crops vs all 9 generated central 128^3 crops.
