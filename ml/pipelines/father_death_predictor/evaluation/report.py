"""Generate final evaluation report."""

import json
import os
from datetime import datetime


def generate_report(results: dict, output_dir: str) -> str:
    """Generate a text report from all evaluation results.

    Args:
        results: dict containing all metric results from the pipeline
        output_dir: directory to save report files

    Returns:
        Report text string
    """
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append("=" * 70)
    lines.append("FATHER DEATH PREDICTION — EVALUATION REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("=" * 70)

    # Dataset info
    if 'dataset' in results:
        ds = results['dataset']
        lines.append(f"\nDataset: train={ds.get('n_train', '?')}, "
                     f"val={ds.get('n_val', '?')}, test={ds.get('n_test', '?')}")
        lines.append(f"Classes: {ds.get('class_dist', '?')}")

    # Baselines
    if 'baselines' in results:
        lines.append("\n--- BASELINES ---")
        for b in results['baselines']:
            lines.append(f"  {b['name']:20s}  F1(macro)={b['f1_macro']:.4f}")

    # Ablation
    if 'ablation' in results:
        lines.append("\n--- ABLATION STUDY ---")
        for name, a in sorted(results['ablation'].items()):
            lines.append(
                f"  {name:15s}  F1={a['f1_macro']:.4f}  "
                f"Kappa={a['kappa']:.4f}  (n_feat={a['n_features']})"
            )

    # Full model
    if 'classifier' in results:
        c = results['classifier']
        lines.append(f"\n--- TUNED CLASSIFIER ---")
        lines.append(f"  F1(macro): {c['f1_macro']:.4f}")
        lines.append(f"  Best params: {c.get('best_params', '?')}")

    # Survival
    if 'survival' in results:
        s = results['survival']
        lines.append(f"\n--- SURVIVAL MODEL (Cox PH) ---")
        lines.append(f"  C-index (train): {s['c_index_train']:.4f}")
        lines.append(f"  C-index (val):   {s['c_index_val']:.4f}")

    # McNemar
    if 'mcnemar' in results:
        m = results['mcnemar']
        lines.append(f"\n--- MCNEMAR TEST vs Demographic Baseline ---")
        lines.append(f"  chi2={m['chi2']:.4f}, p={m['p_value']:.6f}")

    # SHAP top features
    if 'shap_top' in results:
        lines.append("\n--- TOP SHAP FEATURES ---")
        for cls, feats in results['shap_top'].items():
            lines.append(f"  {cls}:")
            for fname, val in feats[:5]:
                lines.append(f"    {fname:30s} |SHAP|={val:.4f}")

    # Yoga reconstruction
    if 'yoga' in results:
        y = results['yoga']
        lines.append(f"\n--- YOGA RECONSTRUCTION ---")
        lines.append(f"  Coverage: {y['coverage']:.1%} of confident alpayu preds match a classical yoga")
        lines.append(f"  Novel patterns: {y['novel_patterns']}")

    # Decision gates
    lines.append("\n--- DECISION GATES ---")
    if 'gates' in results:
        for gate, status in results['gates'].items():
            lines.append(f"  {gate}: {status}")

    # Bootstrap CI
    if 'bootstrap' in results:
        b = results['bootstrap']
        lines.append(f"\n--- BOOTSTRAP 95% CI ---")
        lines.append(f"  F1 mean={b['mean']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")

    lines.append("\n" + "=" * 70)
    report_text = "\n".join(lines)

    # Save
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    # Also save structured JSON
    json_path = os.path.join(output_dir, 'results.json')
    serializable = _make_serializable(results)
    with open(json_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    return report_text


def _make_serializable(obj):
    """Strip non-serializable objects (models, arrays) for JSON output."""
    if isinstance(obj, dict):
        return {
            k: _make_serializable(v) for k, v in obj.items()
            if k not in ('model', 'y_pred', 'y_prob', 'summary')
        }
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj
