# Continuum phase-field model of TDP-43 LCD condensates with CK1δ and calcineurin

Code and data for the continuum model in our preprint
([bioRxiv, doi:10.64898/2026.10.27.762139](https://www.biorxiv.org/content/10.64898/2026.10.27.762139v1)):
Fig. 4a–e and the Supplementary Figures on the continuum model. The molecular-dynamics part
of the study is not included.

```
code/      model (model.py), parameters of Table 1 (config.py), simulation and plotting scripts
data/      simulation output (.npz) the figures are plotted from
figures/   the figures, as produced by the scripts below
```

**Requirements:** Python 3.11, NumPy 1.26, SciPy 1.14, Matplotlib 3.9 (nearby versions should
work). Run all scripts from inside `code/`. Every script documents its options in `--help`.

Model units: L\* = 85.93 nm, T\* = 0.6003 µs. All runs use `config.py` unchanged.

## Figures

| figure | plot script | data | simulation script |
|---|---|---|---|
| Fig. 4e, state diagram | `plot_final_phase.py` | `phase_diagram/phase_diagram.npz` | `scan_phase.py` → `merge_phase.py` → `make_phase_diagram.py` |
| Fig. 4a–d, traces and phase portraits | `plot_md_cells.py`, `plot_md_portraits.py` | `md_points/cell_nk*_np*.npz` | `run_md_fine.py` |
| SI, active vs. passive model | `plot_passive_1d.py` | `passive/*.npz` | `passive_1d.py` |
| SI, initial-condition independence | `plot_init_md.py` | `initial_conditions/run_*.npz` | `scan_init_md.py` |

To regenerate all figures from `data/` (takes minutes):

```bash
cd code
for s in plot_final_phase plot_md_cells plot_md_portraits plot_passive_1d plot_init_md; do python3 $s.py; done
python3 plot_md_portraits.py --panels      # small-multiples version
```

## Regenerating the data

Each 90 µs run (T = 150 model units) takes about 4 h on one core. The scripts run several
simulations in parallel with `multiprocessing`.

```bash
cd code
# state diagram, 14 x 19 (N_k, N_p) cells from the condensate slab
python3 scan_phase.py --nproc 32 --T 150 --c0K -1.6 --retard 50 --pmax 20 \
        --N_k 1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5 \
        --N_p 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 --out phase_slab.npz
python3 merge_phase.py phase_slab*.npz --out phase_slab_merged.npz   # if run in shards
python3 make_phase_diagram.py phase_slab_merged.npz

# traces at the six MD set-ups where the model oscillates
python3 run_md_fine.py --nsave 4000 --nproc 6 --save-dir ../data/md_points

# passive controls (30 µs each)
python3 passive_1d.py --mode noreact
python3 passive_1d.py --mode undriven --Nk 3 --Np 15

# 16 initial condensates at each of the six set-ups (restarts read data/md_points)
python3 scan_init_md.py --procs 32 --T 150
```

## Data format

All files are NumPy `.npz` archives; load with `np.load(f, allow_pickle=True)`. Times `t` are
in model units (× 0.6003 for µs) unless named `*_us`. Periods are in µs.

- `phase_diagram.npz`: per cell `N_k`, `N_p`, `state` (`oscillating` / `steady` / `dissolved`),
  `period`, `pbar` (mean pSer fraction), and the MD set-ups `md_osc`, `md_steady`, `md_dissolved`.
- `cell_nk<K>_np<P>.npz`: `t`, condensate mass `mass`, mean pSer fraction `pfrac` (× 24 = pSer
  per chain), fields `phi0`, `phi1`, `phiK`, `phiP` (frames × 151), `period`, parameters `P_*`.
- `passive_1d_<mode>.npz`: `t_us`, condensed chains `chains`, `pser` per chain, free energy `F`,
  field snapshots `frames` at `t_frames_us` on `x_nm`.
- `run_nk<K>_np<P>_<start>.npz`: as `cell_*`, plus the initial state `x_init` and the
  classification `cls`, `period`, `amp`, `pbar`. Start labels are defined in `scan_init_md.starts()`.

## Citation

If you use this code, please cite the preprint:
https://doi.org/10.64898/2026.10.27.762139

## License

MIT, see `LICENSE`.
