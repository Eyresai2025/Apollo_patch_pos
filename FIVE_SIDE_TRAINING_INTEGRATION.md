# Apollo Five-Side Training Integration

The New SKU Training tab supports both existing individual-side training and the
AI-team one-click five-side training cycle.

## Permanent storage contract

Both training modes use the same model and artifact directories. No model or
side-specific report is stored under a separate `five_side_cycle/runs` tree.

```text
media/training/<SKU>/
├── five_side_training_summary.json
├── five_side_training_summary.csv
├── main_training_config.json
├── sidewall1/
│   ├── <SKU>_sidewall1_patchcore_model.pth
│   ├── training_run_config.json                 # individual run, when used
│   ├── training_timings.csv
│   ├── preprocess_report.json
│   ├── raw_to_patchcore_training_summary.json
│   ├── five_side_training_result.json           # five-side run
│   ├── five_side_worker.log                     # five-side run
│   ├── sidewall_r_anchors.json
│   └── prepared_training/
├── sidewall2/
│   └── same sidewall artifacts
├── tread/
│   ├── <SKU>_tread_patchcore_model.pth
│   ├── training_timings.csv
│   ├── preprocess_report.json
│   ├── tread_raw_to_patchcore_training_summary.json
│   ├── five_side_training_result.json
│   ├── five_side_worker.log
│   └── prepared_training/
├── innerwall/
│   ├── <SKU>_innerwall_patchcore_model.pth
│   ├── training_timings.csv
│   ├── preprocess_report.json
│   ├── inner_raw_to_patchcore_training_summary.json
│   ├── five_side_training_result.json
│   ├── five_side_worker.log
│   └── prepared_training/
└── bead/
    ├── <SKU>_bead_patchcore_model.pth
    ├── training_timings.csv
    ├── preprocess_report.json
    ├── bead_raw_to_patchcore_training_summary.json
    ├── five_side_training_result.json
    ├── five_side_worker.log
    └── prepared_training/
```

The model paths are therefore identical whether the operator trains one side or
uses **Train All 5 Sides**. Existing inference/model-loading code can continue
to load `media/training/<SKU>/<role>/<SKU>_<role>_patchcore_model.pth`.

## Execution order

1. Sidewall 1 and Sidewall 2 run first.
2. Their successful preprocessing reports generate reusable R anchors.
3. Tread, Inner and Bead run using the selected sidewall anchors.
4. Every job writes directly to its own permanent role directory.
