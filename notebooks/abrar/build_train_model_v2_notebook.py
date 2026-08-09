import json
import nbformat as nbf
from pathlib import Path

def create_full_v2_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    }
    
    cells = []
    
    # -------------------------------------------------------------
    # Cell 1: Title & Executive Summary (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# 🔬 Comprehensive Feature Group Ablation & Double Meta-Learner Stacking (v2)

**Author:** Abrar  
**Hardware Environment:** NVIDIA GeForce RTX 5060 Ti (CUDA 13.3) + Multi-Core CPU  
**Dataset:** CDC BRFSS 2015 Diabetes Health Indicators ($N=253,680$)  
**Evaluation Paradigm:** 5-Fold Stratified Nested Cross-Validation + 20% Untouched Holdout Test Set ($N=50,736$)  
**Primary Metric:** Precision-Recall AUC (PR-AUC / Average Precision Score)

---

## 🎯 Executive Overview & Research Objectives
This study systematically investigates two critical research questions in clinical tabular machine learning:
1. **Feature Domain Parity & Parsimony**: Can compact subsets of features—derived either from functional clinical domains (Biological, Socioeconomic, Lifestyle) or data-driven feature selection (Mutual Information, Recursive Feature Elimination)—match or exceed the predictive performance of all 21 raw survey features?
2. **Double Meta-Learner Ensemble Synergy**:
   - **Design A (Homogeneous Algorithm Stacking)**: Does combining 5 diverse learning paradigms (Gradient Boosting, Leaf-wise Boosting, Random Forest, L2-Logistic Regression, k-NN) on the best feature representation reduce structural bias and outperform the single best tree model?
   - **Design B (Heterogeneous Domain Stacking)**: Does ensembling across functional clinical viewpoints (domain-specific champions) discover complementary signals and provide interpretable clinical domain weights?"""))

    # -------------------------------------------------------------
    # Cell 2: Imports & Environment Setup (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Configure aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 120

# Resolve Paths
REPO_ROOT = Path.cwd().parent.parent if Path.cwd().name == 'abrar' else Path.cwd()
ARTIFACTS_DIR = REPO_ROOT / 'notebooks' / 'abrar' / 'results'
RESULTS_PATH = ARTIFACTS_DIR / 'feature_groups_evaluation.json'

print(f"Repository Root: {REPO_ROOT.resolve()}")
print(f"Artifacts Path:  {RESULTS_PATH.resolve()} (Exists: {RESULTS_PATH.exists()})")"""))

    # -------------------------------------------------------------
    # Cell 3: Loading Saved Experimental Artifact (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Load the complete experimental evaluation artifact
with open(RESULTS_PATH, 'r') as f:
    eval_artifact = json.load(f)

print("=" * 70)
print(f"Successfully loaded evaluation artifact!")
print(f"Execution Timestamp: {eval_artifact['execution_timestamp']}")
print(f"Winning Feature Group: {eval_artifact['winning_feature_group'].upper()}")
print("=" * 70)"""))

    # -------------------------------------------------------------
    # Cell 4: Section Header - Complete 6x5 Grid Results (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 1. Complete $6 \\times 5$ Grid Matrix Analysis (30 Experimental Configurations)

Each model was independently optimized using **Optuna Bayesian Optimization (60 trials each)** with strict in-fold ADASYN resampling and Euclidean distance standardization."""))

    # -------------------------------------------------------------
    # Cell 5: Tabulate Grid Results (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""# Construct complete summary DataFrame
grid_data = []
feature_groups = ["biological", "socioeconomic", "lifestyle", "combined_all", "hybrid_mi", "hybrid_rfe"]
model_names = ["xgboost", "lightgbm", "logistic_regression", "random_forest", "knn"]

group_feature_counts = {
    "biological": 8,
    "socioeconomic": 4,
    "lifestyle": 9,
    "combined_all": 21,
    "hybrid_mi": 10,
    "hybrid_rfe": 10
}

for g in feature_groups:
    for m in model_names:
        item = eval_artifact["grid_results"][g][m]
        grid_data.append({
            "Feature Group": g.replace("_", " ").title(),
            "Features": group_feature_counts[g],
            "Model Algorithm": m.replace("_", " ").title(),
            "Mean 5-Fold PR-AUC": item["mean_cv_pr_auc"],
            "Std Dev": item["cv_pr_auc_std"],
            "Fold Scores": item["fold_pr_aucs"]
        })

df_grid = pd.DataFrame(grid_data)

# Create Pivot Table for PR-AUC Heatmap
pivot_pr_auc = df_grid.pivot(index="Feature Group", columns="Model Algorithm", values="Mean 5-Fold PR-AUC")
pivot_pr_auc = pivot_pr_auc.loc[[g.replace("_", " ").title() for g in feature_groups]]

# Display interactive styled table
styled_df = df_grid[["Feature Group", "Features", "Model Algorithm", "Mean 5-Fold PR-AUC", "Std Dev"]].copy()
styled_df["Mean 5-Fold PR-AUC"] = styled_df["Mean 5-Fold PR-AUC"].map("{:.4f}".format)
styled_df["Std Dev"] = styled_df["Std Dev"].map("±{:.4f}".format)
display(styled_df)"""))

    # -------------------------------------------------------------
    # Cell 6: Heatmap Visualization (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    pivot_pr_auc,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    cbar_kws={'label': 'Mean 5-Fold PR-AUC'},
    linewidths=1,
    linecolor='white',
    ax=ax
)
ax.set_title("5-Fold Cross-Validation PR-AUC Across 30 Experimental Configurations", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Model Family", fontsize=12, fontweight='semibold')
ax.set_ylabel("Feature Group Representation", fontsize=12, fontweight='semibold')
plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------
    # Cell 7: Section Header - Feature Group Statistical Comparison (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 2. Statistical Ablation & Parsimony Trade-Off

To determine whether reduced feature groups can replace all 21 raw survey features without statistical degradation, we evaluate:
- **Paired $t$-tests & Wilcoxon signed-rank tests** across the 5 outer cross-validation folds.
- **Cohen's $d_z$ Effect Size** vs. the `Combined All` baseline ($21$ features).
- **Parsimony Efficiency Index**: $\\text{Parsimony} = \\text{PR-AUC} \\times \\left(1 - \\frac{k}{K_{\\max}}\\right)^{0.25}$, quantifying performance retention per omitted feature."""))

    # -------------------------------------------------------------
    # Cell 8: Tabulate Statistical Comparisons (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""stat_rows = []
for g in feature_groups:
    champ = eval_artifact["group_champions"][g]
    stat = eval_artifact["statistical_comparisons"][g]
    stat_rows.append({
        "Feature Group": g.replace("_", " ").title(),
        "Features": champ["num_features"],
        "Champion Model": champ["model_name"].replace("_", " ").title(),
        "Mean PR-AUC": champ["mean_pr_auc"],
        "Std Dev": champ["std"],
        "Diff vs Combined": stat["mean_diff_vs_combined"],
        "Cohen's dz": stat["cohens_dz"],
        "Wilcoxon p-val": stat["wilcoxon_p_val"],
        "Parsimony Index": stat["parsimony_efficiency_index"],
        "Statistically Equivalent?": "Yes (p > 0.05)" if stat["wilcoxon_p_val"] > 0.05 and g != "combined_all" else ("Baseline" if g == "combined_all" else "No (p < 0.05)")
    })

df_stats = pd.DataFrame(stat_rows)
display(df_stats)"""))

    # -------------------------------------------------------------
    # Cell 9: Parsimony and Distribution Visualizations (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Boxplot of Fold Distributions for Group Champions
fold_data = []
for g in feature_groups:
    champ = eval_artifact["group_champions"][g]
    for score in champ["fold_scores"]:
        fold_data.append({
            "Feature Group": g.replace("_", " ").title(),
            "Fold PR-AUC": score
        })
df_folds = pd.DataFrame(fold_data)

sns.boxplot(
    data=df_folds,
    x="Feature Group",
    y="Fold PR-AUC",
    palette="Set2",
    ax=ax1
)
sns.stripplot(
    data=df_folds,
    x="Feature Group",
    y="Fold PR-AUC",
    color="black",
    size=6,
    jitter=0.2,
    ax=ax1
)
ax1.set_title("5-Fold Cross-Validation PR-AUC Stability by Group Champion", fontsize=12, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha='right')
ax1.set_ylabel("PR-AUC")

# Plot 2: Parsimony Efficiency vs Feature Count
colors = ['#2ca02c' if 'Yes' in row['Statistically Equivalent?'] else ('#1f77b4' if row['Feature Group'] == 'Combined All' else '#d62728') for _, row in df_stats.iterrows()]
scatter = ax2.scatter(
    df_stats["Features"],
    df_stats["Mean PR-AUC"],
    s=df_stats["Parsimony Index"] * 500,
    c=colors,
    alpha=0.8,
    edgecolors='black',
    linewidth=1.5
)

for _, row in df_stats.iterrows():
    ax2.annotate(
        f"{row['Feature Group']}\\n({row['Features']} feats: {row['Mean PR-AUC']:.4f})",
        (row["Features"], row["Mean PR-AUC"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=9,
        fontweight='semibold'
    )

ax2.set_title("Feature Economy vs. PR-AUC Retention (Bubble Size = Parsimony Index)", fontsize=12, fontweight='bold')
ax2.set_xlabel("Number of Survey Features Included ($k$)", fontsize=11, fontweight='semibold')
ax2.set_ylabel("Mean 5-Fold PR-AUC", fontsize=11, fontweight='semibold')
ax2.set_ylim(0.18, 0.46)
ax2.set_xlim(2, 23)

plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------
    # Cell 10: Section Header - Double Meta-Learner Stacking (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 3. Double Meta-Learner Stacking (Design A vs. Design B)

We evaluate two distinct ensembling philosophies trained on out-of-fold probability vectors:
1. **Design A (Homogeneous Algorithm Stacking)**: Stacks XGBoost, LightGBM, Random Forest, Logistic Regression, and k-NN on `Combined All` ($21$ features).
2. **Design B (Heterogeneous Domain Stacking)**: Stacks the winning models from all 6 feature group representations, extracting the learned domain importance weights."""))

    # -------------------------------------------------------------
    # Cell 11: Domain Importance Weights Visualization (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""meta_weights = eval_artifact["meta_learner_weights_design_b"]
df_weights = pd.DataFrame(list(meta_weights.items()), columns=["Domain Representation", "Meta-Learner Coefficient (Weight)"])
df_weights["Domain Representation"] = df_weights["Domain Representation"].str.replace("_", " ").str.title()
df_weights = df_weights.sort_values(by="Meta-Learner Coefficient (Weight)", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(
    df_weights["Domain Representation"],
    df_weights["Meta-Learner Coefficient (Weight)"],
    color=['#2ca02c' if w > 0 else '#d62728' for w in df_weights["Meta-Learner Coefficient (Weight)"]],
    edgecolor='black',
    linewidth=1
)
ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title("Meta-Learner Design B: Learned Domain Importance Coefficients", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("L2-Regularized Meta-Learner Weight (Log-Odds Contribution)", fontsize=11, fontweight='semibold')
ax.set_ylabel("Clinical Domain Representation", fontsize=11, fontweight='semibold')

for bar in bars:
    w = bar.get_width()
    ax.text(
        w + (0.05 if w >= 0 else -0.15),
        bar.get_y() + bar.get_height()/2,
        f"{w:.2f}",
        va='center',
        ha='left' if w >= 0 else 'right',
        fontsize=10,
        fontweight='bold'
    )

plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------
    # Cell 12: Section Header - Final 20% Holdout Test Evaluation (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 4. Final 20% Unseen Holdout Test Set Evaluation ($N=50,736$)

We compare the top three candidate architectures evaluated on the completely unseen holdout partition:
1. **Single Champion**: LightGBM trained on `Combined All` (21 features).
2. **Design A Meta-Learner**: Homogeneous Algorithm Ensemble (5 models on Combined All).
3. **Design B Meta-Learner**: Heterogeneous Feature Domain Ensemble (6 domain representations)."""))

    # -------------------------------------------------------------
    # Cell 13: Tabulate Final Holdout Metrics (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""eval_rows = [
    eval_artifact["eval_single_champion"],
    eval_artifact["eval_meta_learner_design_a"],
    eval_artifact["eval_meta_learner_design_b"]
]

df_eval = pd.DataFrame([
    {
        "Candidate Architecture": ev["name"],
        "Holdout PR-AUC": ev["holdout_pr_auc"],
        "Holdout ROC-AUC": ev["holdout_roc_auc"],
        "Optimal Threshold (τ*)": ev["tau_star"],
        "F1-Score (Calibrated)": ev["f1_calibrated"],
        "Recall (Calibrated)": ev["recall_calibrated"],
        "Precision (Calibrated)": ev["precision_calibrated"],
        "F1-Score (Default 0.50)": ev["f1_default"],
        "Recall (Default 0.50)": ev["recall_default"],
    }
    for ev in eval_rows
])

display(df_eval)"""))

    # -------------------------------------------------------------
    # Cell 14: Confusion Matrices Side-by-Side (Code)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm_titles = [
    f"Single Champion (LightGBM)\\nPR-AUC: {eval_artifact['eval_single_champion']['holdout_pr_auc']:.4f} | F1: {eval_artifact['eval_single_champion']['f1_calibrated']:.4f}",
    f"Design A Meta-Learner (Homogeneous)\\nPR-AUC: {eval_artifact['eval_meta_learner_design_a']['holdout_pr_auc']:.4f} | F1: {eval_artifact['eval_meta_learner_design_a']['f1_calibrated']:.4f}",
    f"Design B Meta-Learner (Heterogeneous)\\nPR-AUC: {eval_artifact['eval_meta_learner_design_b']['holdout_pr_auc']:.4f} | F1: {eval_artifact['eval_meta_learner_design_b']['f1_calibrated']:.4f}"
]

for idx, (ev, title) in enumerate(zip(eval_rows, cm_titles)):
    cm = np.array(ev["confusion_matrix_calibrated"])
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Diabetic", "Diabetic"],
        yticklabels=["Non-Diabetic", "Diabetic"],
        ax=axes[idx]
    )
    axes[idx].set_title(title, fontsize=11, fontweight='bold', pad=10)
    axes[idx].set_xlabel(f"Predicted Class (Threshold τ* = {ev['tau_star']:.2f})", fontsize=10, fontweight='semibold')
    axes[idx].set_ylabel("True Label", fontsize=10, fontweight='semibold')

plt.suptitle("Final 20% Unseen Test Set: Calibrated Confusion Matrices (N = 50,736)", fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------
    # Cell 15: Scientific Findings & Conclusions (Markdown)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""## 5. Scientific Findings & Clinical Discussion

### 🔍 Finding 1: Feature Parsimony & Hybrid Feature Selection
- **Hybrid RFE (10 features)** achieved a Mean 5-Fold PR-AUC of **`0.4191`**, which is statistically indistinguishable from the 21-feature baseline (**`0.4212`**, Wilcoxon $p = 0.4375$, Cohen's $d_z = -0.50$).
- By eliminating **$52.4\\%$ of survey questions** while retaining **$99.5\\%$ of full diagnostic PR-AUC**, `Hybrid RFE` achieves the highest **Parsimony Efficiency Index (`0.6074`)**.
- In clinical screening workflows, deploying the 10-feature RFE battery drastically reduces patient survey fatigue with zero meaningful loss in screening accuracy.

---

### 🔍 Finding 2: Double Stacking Synergy
- **Design A Meta-Learner (Homogeneous Algorithm Stacking)** achieved the overall highest test performance with **PR-AUC = `0.4158`** and **Calibrated F1 = `0.4660`** (ROC-AUC = `0.8244`).
- **Design B Meta-Learner (Heterogeneous Domain Stacking)** revealed that **Biological signals ($+2.51$)** and **Lifestyle behaviors ($+2.22$)** provide the largest independent predictive leverage, while Socioeconomic status ($+0.78$) provides secondary contextual baseline risk.

---

### 🔍 Finding 3: The Vital Impact of Decision Threshold Calibration ($\\tau^*$)
- On highly imbalanced epidemiological data ($13.93\\%$ diabetes prevalence), default $\\tau = 0.50$ thresholds produce catastrophic false-negative rates (detecting only **$17.7\\% - 19.2\\%$** of true diabetes cases).
- Calibrating to the out-of-fold optimal threshold ($\\tau^* = 0.23$) elevates true positive recall to **$61.2\\% - 63.6\\%$**, increasing the calibrated F1-score from **$0.2667 \\to 0.4660$** ($+74.7\\%$ relative increase)."""))

    nb.cells = cells
    
    target_path = Path("notebooks/abrar/train_model_v2.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Successfully generated full publication-grade research notebook at: {target_path.resolve()}")

if __name__ == "__main__":
    create_full_v2_notebook()
