# Adjudication Kernel — Reproduction Artifact (anonymous)

This repository is released **anonymously** for double-blind review of the
accompanying manuscript (*Adjudicate, Don't Gate: A Classical-Physics
Reasoning Kernel for Physical AI*). Author identity is withheld until acceptance.

## Layout
- `data/` — all `_audit_e*.json` result files and `dataset_vlm_ood_manifest.json`.
  These are the **sole source** of every numeric claim in the paper.
- `verifier/` — `_audit_e*.py` (per-experiment self-consistency checks) and
  `blind_*.py` (independent blind replication scripts). Each reads its sibling
  JSON from `data/`; run with `python verifier/<script>.py`.

## E-id -> file mapping
(Every headline number traces to one of these.)
| Experiment | Result JSON |
|---|---|
| E56 | _audit_e56.json, _audit_e56c.json |
| E57 | _audit_e57d.json |
| E58 | _audit_e58.json, _audit_e58_replicator_anomaly.json, _audit_e58_strict.json |
| E59 | _audit_e59_replicator_anomaly.json, _audit_e59_strict.json, _audit_e59b_strict.json |
| E60 | _audit_e60_replicator_anomaly.json, _audit_e60_strict.json, _audit_e60b_strict.json |
| E64 | _audit_e64_strict.json |
| E65 | _audit_e65_replicator_anomaly.json, _audit_e65_strict.json, _audit_e65b_strict.json |
| E66 | _audit_e66.json, _audit_e66_replicator_anomaly.json, _audit_e66_strict.json |
| E67 | _audit_e67_strict.json, _audit_e67_testbed.json |
| E68 | _audit_e68_control.json, _audit_e68_replicator_anomaly.json, _audit_e68_strict.json |
| E69 | _audit_e69_bandwidth.json, _audit_e69_replicator_anomaly.json, _audit_e69_strict.json |
| E70 | _audit_e70_replicator_anomaly.json, _audit_e70_residual.json, _audit_e70_strict.json |
| E71 | _audit_e71_gain_budget.json, _audit_e71_replicator_anomaly.json, _audit_e71_strict.json |
| E72 | _audit_e72_friction_cone.json, _audit_e72_replicator_anomaly.json, _audit_e72_strict.json |
| E73 | _audit_e73_replicator_anomaly.json, _audit_e73_strict.json, _audit_e73_velocity_spectrum.json |
| E74 | _audit_e74_bandwidth.json, _audit_e74_replicator_anomaly.json, _audit_e74_strict.json |
| E75 | _audit_e75_energy.json, _audit_e75_replicator_anomaly.json, _audit_e75_strict.json |
| E76 | _audit_e76_coordination.json, _audit_e76_replicator_anomaly.json, _audit_e76_strict.json |
| E77 | _audit_e77_controller.json, _audit_e77_replicator_anomaly.json, _audit_e77_strict.json |
| E79 | _audit_e79_replicator_anomaly.json, _audit_e79_strict.json |
| E80 | _audit_e80_replicator_anomaly.json, _audit_e80_strict.json |
| E81 | _audit_e81_replicator_anomaly.json, _audit_e81_strict.json |
| E82 | _audit_e82_replicator_anomaly.json, _audit_e82_strict.json |
| E83 | _audit_e83_replicator_anomaly.json, _audit_e83_strict.json |
| E84 | _audit_e84_replicator_anomaly.json, _audit_e84_strict.json, _audit_e84b_replicator_anomaly.json, _audit_e84b_strict.json, _audit_e84c_replicator_anomaly.json, _audit_e84c_strict.json |
| E85 | _audit_e85_replicator_anomaly.json, _audit_e85_strict.json, _audit_e85b_replicator_anomaly.json, _audit_e85b_strict.json |
| E86 | _audit_e86_replicator_anomaly.json, _audit_e86_strict.json |

## Reproduce a check
```
cd verifier
python _audit_e56.py      # reads ../data/_audit_e56.json, prints PASS/FAIL
```

## Data availability
The files above are sufficient to **independently reproduce and verify every
reported number**. Additional raw experiment traces / intermediate artifacts
(large, not required for verification) are available from the authors upon
reasonable request after acceptance.

## License
Code: MIT. Data: CC-BY-NC. See LICENSE.
