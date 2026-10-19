# Individual loss-model functions

All calculations import the current production loss helpers. The shock coordinate tends to 0.3 below its 0.5 join, shock loss tends to 0.7, and the Reynolds multiplier tends to 0.457305052. Loading preserves physical turning and continues only the mean secant from sec80 degrees toward sec87.5 degrees; secondary projection uses physical cosines. Values and derivatives are saved separately from plotting.

## Reports

- [Modeling choices and numerical behavior](report/00_ko_safeguards_overview.md)
- [KO profile, shock and Reynolds correction](report/01_ko_profile.md)
- [KO secondary loss](report/02_ko_secondary.md)
- [KO trailing-edge loss](report/03_ko_trailing.md)
- [KO clearance loss](report/04_ko_clearance.md)
- [Loss-coefficient conversions](report/05_conversions.md)
- [Shared mathematics](report/06_shared_math.md)
- [Moustapha incidence](report/07_moustapha_incidence.md)
- [Moustapha secondary factor](report/08_moustapha_secondary.md)
- [Benner secondary loss](report/09_benner_secondary.md)
- [Benner incidence](report/10_benner_incidence.md)
- [Benner penetration](report/11_benner_penetration.md)

## Reproduction

```powershell
python -B development/loss_model_analysis/loss_model_audit/individual_functions/scripts/run.py
python -B development/loss_model_analysis/loss_model_audit/individual_functions/scripts/render_saved.py
```

[Methods](methodology.md) | [Current numerical verification](data/validation/reproduction.json) | [Model source hashes](data/provenance/source_manifest.json) | [Shock analysis](../../kacker_okapuu_shock/readme.md) | [Stage analyses](../overall_stage/readme.md)
