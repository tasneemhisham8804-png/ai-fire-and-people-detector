"""
Empirical evaluation harness for the FULL pipeline (fire classifier + people
detector) against real, human-labeled video.

WHAT THIS DOES
--------------
Given a labeled manifest of video clips, runs each one through the same
run_analysis() logic /analyze uses (in-process, no HTTP needed) and reports
precision, recall, F1, and accuracy for both the fire detector and the
people detector against ground truth.

WHY THIS EXISTS / WHAT IT DOESN'T CLAIM
----------------------------------------
The unit tests in this project (test_fire_detector.py, test_people_detector.py)
only confirm pure, model-free logic behaves correctly on synthetic inputs.
evaluate_fire_localizer_synthetic.py goes a step further for the classical-CV
localizer specifically, with real synthetic precision/recall numbers. But
NONE of that says anything about how well the actual trained models
(Fire_DETECTOR.keras, best.pt) perform on real footage — that can only be
measured against real, held-out, human-labeled video, which requires:
  1. The trained model weight files (not included in this repo — see README)
  2. A labeled set of test clips this project does not currently ship with

This script is the tool for running that evaluation once both exist. It is
NOT usable against the conftest.py stubs (those stub out inference entirely,
which is the opposite of what an accuracy evaluation needs), and running it
without real weights/labeled clips will simply fail — that's intentional,
not a bug.

MANIFEST FORMAT (CSV)
----------------------
video_path,fire_label,people_label
clips/fire_01.mp4,1,1
clips/fire_02.mp4,1,0
clips/no_fire_01.mp4,0,0
...
fire_label / people_label are 1 (present) or 0 (absent), assigned by a human
reviewing the clip directly — NOT by running the model on it first (that
would be evaluating the model against its own output).

USAGE
-----
    python evaluate.py --manifest labeled_clips.csv

Recommended starting point: 20-30 clips covering both classes for each
label, plus some clips designed to stress the known failure modes in
LIMITATIONS.md (night footage, distant/small fire, partially-occluded
people, warm-lit skin close to camera).
"""
import argparse
import csv
import sys
from pathlib import Path


def _confusion_counts(labels_and_preds):
    tp = fp = tn = fn = 0
    for label, pred in labels_and_preds:
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1
    return tp, fp, tn, fn


def _metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and precision == precision and recall == recall
          else float("nan"))
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else float("nan")
    return {
        "precision": round(precision, 3) if precision == precision else precision,
        "recall": round(recall, 3) if recall == recall else recall,
        "f1": round(f1, 3) if f1 == f1 else f1,
        "accuracy": round(accuracy, 3) if accuracy == accuracy else accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def run_evaluation(manifest_path: Path):
    # Imported here, not at module level, so this script's own --help works
    # even without the real model dependencies installed — the import (and
    # the failure if models aren't loaded) only happens once an actual
    # evaluation run is attempted.
    from MAIN import run_analysis, fire_model

    if fire_model is None:
        print(
            "ERROR: fire_model is None — the trained model didn't load. "
            "This script measures real model accuracy, so it can't run "
            "against conftest.py's stubs or a missing weights file. "
            "Confirm Fire_DETECTOR.keras is present and loadable (see README setup).",
            file=sys.stderr,
        )
        sys.exit(1)

    fire_results = []
    people_results = []

    with manifest_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = Path(row["video_path"])
            if not video_path.exists():
                print(f"WARNING: {video_path} not found, skipping", file=sys.stderr)
                continue

            result = run_analysis(video_path)

            fire_label = int(row["fire_label"])
            people_label = int(row["people_label"])
            fire_pred = int(result["fire_detected"])
            people_pred = int(result["people_detected"])

            fire_results.append((fire_label, fire_pred))
            people_results.append((people_label, people_pred))

            print(f"{video_path.name}: fire label={fire_label} pred={fire_pred} | "
                  f"people label={people_label} pred={people_pred} | verdict={result['verdict']}")

    if not fire_results:
        print("No clips were evaluated — check that video_path values in the manifest resolve to real files.",
              file=sys.stderr)
        sys.exit(1)

    print("\n=== Fire detection ===")
    for k, v in _metrics(*_confusion_counts(fire_results)).items():
        print(f"  {k}: {v}")

    print("\n=== People detection ===")
    for k, v in _metrics(*_confusion_counts(people_results)).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fire/people detection accuracy against real, human-labeled video clips."
    )
    parser.add_argument(
        "--manifest", required=True, type=Path,
        help="CSV with columns: video_path,fire_label,people_label (see this file's module docstring for format)",
    )
    args = parser.parse_args()
    run_evaluation(args.manifest)
