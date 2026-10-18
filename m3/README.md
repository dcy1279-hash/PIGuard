# FIT5230 M3: Unicode attack experiments

This directory contains our M3 source, public benchmark data and frozen cohort definitions. M1 and M2 files elsewhere in the repository are unchanged.

## Run in Colab

Use the M3 notebook shared for assessment, select a T4 GPU, and run top to bottom. Its setup cell downloads this directory at a fixed Git commit and verifies every required file against `inventory.json`.

- Default: actual PIGuard model loading and example predictions, combination controls, and six matched Unicode-family methods on both detectors.
- Optional, OFF by default: gradient ranking, paid-probe selection, full stability/repair, and direct transfer. Switches are in Section 4.1; direct transfer also requires stability.
- Model weights are downloaded separately. No model weights, private credentials or completed result archives are included here.
- The frozen plans define previously explored cohorts; they are not independent holdouts. Historical validation metadata is not used as a substitute for current model predictions.
- Conclusions and figures are calculated from the enabled, completed experiments in the current run. Save the Colab outputs before submission. Downloading an evidence ZIP is optional.

## Files

- `src/`: unchanged model/protocol implementations and the native-run driver.
- `datasets/`: five public benchmark JSON files.
- `plans/`: four fixed cohort/protocol definitions.
- `inventory.json`: SHA256 hashes for the required source, data and plans.
- `LICENSE`: the original repository licence.

The notebook shows the model and core attack implementations, with concise English methods, controls and limitations. This GitHub directory replaces the former encoded setup package; the experiments themselves are unchanged.

Local tests verify source integrity and control flow, not GPU outcomes. Actual model inference still runs in Colab. The original classifier, checkpoint, thresholds and labels are not trained or changed by this work.

Based on [PIGuard](https://github.com/leolee99/PIGuard); the comparison model is `protectai/deberta-v3-base-prompt-injection-v2` at revision `90c9989b1a342275dd0d1a95aad283c04e075671`.
