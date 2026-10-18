# Turbine loss-model analysis

The scripts import the current repository loss functions. Shock uses a shared hub/tip coordinate with lower asymptote 0.3 and upper shock asymptote 0.7. The high-Reynolds correction tends to 0.457305052.

- [Individual functions and equations](individual_functions/readme.md)
- [Side-by-side stage comparison](overall_stage/comparison.md)
- [Full Kacker-Okapuu turbine sensitivities](overall_stage/sensitivity_kacker_okapuu/readme.md)
- [Full Benner turbine sensitivities](overall_stage/sensitivity_benner/readme.md)
- [Isentropic zero-loss turbine sensitivities](overall_stage/sensitivity_isentropic/readme.md)
- [Shock and Reynolds analysis](../kacker_okapuu_shock/readme.md)

Run `python -B loss_model_analysis/loss_model_audit/run.py` to calculate both audit sections. Use `--render-only` to render saved data or `--check` to verify navigation.
