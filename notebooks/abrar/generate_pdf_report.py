import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
    PageBreak
)
from reportlab.pdfgen import canvas

# Configure aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 200

# Paths
REPO_ROOT = Path("d:/ML Project/diabetes-brfss-ml")
ARTIFACTS_DIR = REPO_ROOT / "notebooks" / "abrar" / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "pdf_figures"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = ARTIFACTS_DIR / "feature_groups_evaluation.json"
PDF_PATH = ARTIFACTS_DIR / "diabetes_ablation_and_stacking_report.pdf"

# Load JSON Data
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# ==============================================================================
# 1. GENERATE HIGH-RES CHARTS
# ==============================================================================

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

# --- Plot 1: Heatmap of 6x5 Grid ---
grid_data = []
for g in feature_groups:
    for m in model_names:
        item = data["grid_results"][g][m]
        grid_data.append({
            "Feature Group": g.replace("_", " ").title(),
            "Model": m.replace("_", " ").title(),
            "PR-AUC": item["mean_cv_pr_auc"]
        })
df_grid = pd.DataFrame(grid_data)
pivot = df_grid.pivot(index="Feature Group", columns="Model", values="PR-AUC")
pivot = pivot.loc[[g.replace("_", " ").title() for g in feature_groups]]

fig, ax = plt.subplots(figsize=(7.5, 3.8))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Mean 5-Fold PR-AUC'}, linewidths=0.8, linecolor='white', ax=ax)
ax.set_title("5-Fold Cross-Validation PR-AUC Across 30 Experimental Configurations", fontsize=11, fontweight='bold', pad=8)
ax.set_xlabel("Model Family", fontsize=9, fontweight='semibold')
ax.set_ylabel("Feature Group Representation", fontsize=9, fontweight='semibold')
plt.tight_layout()
heatmap_path = PLOTS_DIR / "fig1_grid_heatmap.png"
plt.savefig(heatmap_path)
plt.close()

# --- Plot 2: Boxplots of Fold PR-AUC & Parsimony ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

fold_data = []
for g in feature_groups:
    champ = data["group_champions"][g]
    for score in champ["fold_scores"]:
        fold_data.append({"Group": g.replace("_", " ").title(), "Score": score})
df_folds = pd.DataFrame(fold_data)

sns.boxplot(data=df_folds, x="Group", y="Score", palette="Set2", ax=ax1)
sns.stripplot(data=df_folds, x="Group", y="Score", color="black", size=4, jitter=0.2, ax=ax1)
ax1.set_title("5-Fold CV PR-AUC Stability (Group Champions)", fontsize=9, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha='right', fontsize=8)
ax1.set_ylabel("PR-AUC", fontsize=8)

stat_rows = []
for g in feature_groups:
    champ = data["group_champions"][g]
    stat = data["statistical_comparisons"][g]
    stat_rows.append({
        "Group": g.replace("_", " ").title(),
        "Features": champ["num_features"],
        "PR-AUC": champ["mean_pr_auc"],
        "Parsimony": stat["parsimony_efficiency_index"]
    })
df_stats = pd.DataFrame(stat_rows)

colors_bubble = ['#1f77b4' if r['Group']=='Combined All' else ('#2ca02c' if r['Group']=='Hybrid Rfe' else '#ff7f0e') for _, r in df_stats.iterrows()]
ax2.scatter(df_stats["Features"], df_stats["PR-AUC"], s=df_stats["Parsimony"]*350, c=colors_bubble, alpha=0.85, edgecolors='black', linewidth=1)
for _, r in df_stats.iterrows():
    ax2.annotate(f"{r['Group']} ({r['Features']})", (r["Features"], r["PR-AUC"]), textcoords="offset points", xytext=(0, 6), ha='center', fontsize=7.5, fontweight='bold')
ax2.set_title("Feature Economy vs. PR-AUC Retention", fontsize=9, fontweight='bold')
ax2.set_xlabel("Number of Survey Features (k)", fontsize=8, fontweight='semibold')
ax2.set_ylabel("Mean 5-Fold PR-AUC", fontsize=8, fontweight='semibold')
ax2.set_ylim(0.18, 0.46)
ax2.set_xlim(2, 23)

plt.tight_layout()
box_parsimony_path = PLOTS_DIR / "fig2_box_parsimony.png"
plt.savefig(box_parsimony_path)
plt.close()

# --- Plot 3: Meta-Learner Design B Domain Weights ---
weights = data["meta_learner_weights_design_b"]
df_w = pd.DataFrame(list(weights.items()), columns=["Domain", "Weight"]).sort_values(by="Weight", ascending=False)
df_w["Domain"] = df_w["Domain"].str.replace("_", " ").str.title()

fig, ax = plt.subplots(figsize=(7.5, 2.6))
bars = ax.barh(df_w["Domain"], df_w["Weight"], color=['#2ca02c' if w > 0 else '#d62728' for w in df_w["Weight"]], edgecolor='black', linewidth=0.8)
ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title("Meta-Learner Design B: Learned Clinical Domain Weights", fontsize=9.5, fontweight='bold', pad=6)
ax.set_xlabel("L2-Regularized Logistic Regression Weight (Log-Odds)", fontsize=8, fontweight='semibold')
for b in bars:
    w = b.get_width()
    ax.text(w + (0.05 if w >= 0 else -0.15), b.get_y() + b.get_height()/2, f"{w:.2f}", va='center', ha='left' if w>=0 else 'right', fontsize=8, fontweight='bold')
plt.tight_layout()
weights_path = PLOTS_DIR / "fig3_domain_weights.png"
plt.savefig(weights_path)
plt.close()

# --- Plot 4: Final Confusion Matrices Side-by-Side ---
eval_rows = [
    data["eval_single_champion"],
    data["eval_meta_learner_design_a"],
    data["eval_meta_learner_design_b"]
]
cm_titles = [
    f"Single Champion (LightGBM)\nPR-AUC: {eval_rows[0]['holdout_pr_auc']:.4f} | F1: {eval_rows[0]['f1_calibrated']:.4f}",
    f"Design A Meta-Learner (Homogeneous)\nPR-AUC: {eval_rows[1]['holdout_pr_auc']:.4f} | F1: {eval_rows[1]['f1_calibrated']:.4f}",
    f"Design B Meta-Learner (Heterogeneous)\nPR-AUC: {eval_rows[2]['holdout_pr_auc']:.4f} | F1: {eval_rows[2]['f1_calibrated']:.4f}"
]

fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
for idx, (ev, t) in enumerate(zip(eval_rows, cm_titles)):
    cm = np.array(ev["confusion_matrix_calibrated"])
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, xticklabels=["Non-Diab", "Diabetic"], yticklabels=["Non-Diab", "Diabetic"], ax=axes[idx], annot_kws={'size': 8})
    axes[idx].set_title(t, fontsize=8, fontweight='bold', pad=6)
    axes[idx].set_xlabel(f"Predicted (τ* = {ev['tau_star']:.2f})", fontsize=7.5, fontweight='semibold')
    axes[idx].set_ylabel("True Label", fontsize=7.5, fontweight='semibold')
plt.tight_layout()
cm_path = PLOTS_DIR / "fig4_confusion_matrices.png"
plt.savefig(cm_path)
plt.close()

print("All charts successfully rendered.")

# ==============================================================================
# 2. BUILD PROFESSIONAL PDF REPORT
# ==============================================================================

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#4B5563"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(40, 760, "CDC BRFSS 2015 Diabetes ML Study | Feature Ablation & Meta-Learner Stacking")
            self.setStrokeColor(colors.HexColor("#D1D5DB"))
            self.setLineWidth(0.5)
            self.line(40, 752, 572, 752)
        
        # Footer
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#D1D5DB"))
        self.setLineWidth(0.5)
        self.line(40, 42, 572, 42)
        self.drawString(40, 30, "Confidential & Proprietary | Machine Learning Research Report")
        self.drawRightString(572, 30, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=44,
    bottomMargin=48
)

styles = getSampleStyleSheet()

# Custom Palette Styles
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    textColor=colors.HexColor("#1E3A8A"),
    spaceAfter=4
)

subtitle_style = ParagraphStyle(
    "DocSubTitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#4B5563"),
    spaceAfter=10
)

h1_style = ParagraphStyle(
    "Heading1_Custom",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    textColor=colors.HexColor("#1E3A8A"),
    spaceBefore=8,
    spaceAfter=4
)

h2_style = ParagraphStyle(
    "Heading2_Custom",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    textColor=colors.HexColor("#0D9488"),
    spaceBefore=6,
    spaceAfter=3
)

body_style = ParagraphStyle(
    "Body_Custom",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#1F2937"),
    spaceAfter=4
)

callout_style = ParagraphStyle(
    "Callout_Text",
    parent=styles["Normal"],
    fontName="Helvetica-Oblique",
    fontSize=8.5,
    leading=11.5,
    textColor=colors.HexColor("#1E293B"),
)

table_header_style = ParagraphStyle(
    "TH",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=7.5,
    leading=9.5,
    textColor=colors.white,
    alignment=1
)

table_cell_style = ParagraphStyle(
    "TC",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=9.5,
    textColor=colors.HexColor("#111827"),
    alignment=1
)

table_cell_left = ParagraphStyle(
    "TCL",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=9.5,
    textColor=colors.HexColor("#111827"),
    alignment=0
)

story = []

# Title & Metadata Banner
story.append(Paragraph("Comprehensive Feature Categorization Ablation & Double Meta-Learner Stacking", title_style))
story.append(Paragraph("<b>Author:</b> Abrar &nbsp;|&nbsp; <b>Dataset:</b> CDC BRFSS 2015 ($N=253,680$) &nbsp;|&nbsp; <b>Primary Metric:</b> PR-AUC (Average Precision)", subtitle_style))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1E3A8A"), spaceAfter=8))

# Executive Summary Callout Box
summary_html = """<b>Executive Summary:</b> This study presents a leak-free machine learning investigation into feature economy and hierarchical ensembling across 30 experimental configurations (6 feature representations × 5 model families, 60 Optuna trials each = 1,800 trials). 
<br/><br/>
<b>Key Finding 1 (Parsimony):</b> <code>Hybrid RFE</code> (10 features) retains <b>99.5%</b> of full diagnostic PR-AUC (<code>0.4191</code> vs <code>0.4212</code>) with no statistically significant difference (Wilcoxon <i>p</i> = 0.4375 > 0.05), reducing clinical survey burden by <b>52.4%</b>.
<br/>
<b>Key Finding 2 (Stacking Synergy):</b> <code>Design A Meta-Learner</code> (Homogeneous Algorithm Stacking) achieved the overall champion holdout score (<b>PR-AUC: 0.4158</b>, <b>ROC-AUC: 0.8244</b>, <b>Calibrated F1: 0.4660</b>). <code>Design B Meta-Learner</code> revealed that Biological (+2.51) and Lifestyle (+2.22) indicators dominate predictive log-odds.
<br/>
<b>Key Finding 3 (Threshold Calibration):</b> Calibrating to optimal decision threshold τ* elevated true diabetes sensitivity from <b>17.7% → 61.2%</b> (+74.7% relative F1 increase)."""

callout_data = [[Paragraph(summary_html, callout_style)]]
t_callout = Table(callout_data, colWidths=[536])
t_callout.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#16A34A")),
    ('TOPPADDING', (0, 0), (-1, -1), 6),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
]))
story.append(t_callout)
story.append(Spacer(1, 8))

# Section 1: Methodology
story.append(Paragraph("1. Leak-Free Methodology & Architecture", h1_style))
story.append(Paragraph(
    "To ensure total empirical integrity, data resampling via <b>Standardized ADASYN</b> was strictly isolated within outer cross-validation training folds ($k-1$ folds). Continuous and ordinal synthetic samples were mapped and clipped to survey integer domains. Bayesian Optimization with 60 trials per model guided hyperparameter convergence across 5 diverse model families (XGBoost GPU, LightGBM, Random Forest, Logistic Regression, k-NN).",
    body_style
))
story.append(Spacer(1, 4))

# Section 2: Full 6x5 Grid Matrix
story.append(Paragraph("2. Complete 6 × 5 Experimental Grid Matrix (Mean 5-Fold PR-AUC)", h1_style))

grid_headers = [
    Paragraph("<b>Feature Group</b>", table_header_style),
    Paragraph("<b>Feats (k)</b>", table_header_style),
    Paragraph("<b>XGBoost (GPU)</b>", table_header_style),
    Paragraph("<b>LightGBM</b>", table_header_style),
    Paragraph("<b>Logistic Reg</b>", table_header_style),
    Paragraph("<b>Random Forest</b>", table_header_style),
    Paragraph("<b>k-NN</b>", table_header_style),
    Paragraph("<b>Group Champion</b>", table_header_style),
]

grid_table_data = [grid_headers]
for g in feature_groups:
    row = [
        Paragraph(g.replace("_", " ").title(), table_cell_left),
        Paragraph(str(group_feature_counts[g]), table_cell_style),
        Paragraph(f"{data['grid_results'][g]['xgboost']['mean_cv_pr_auc']:.4f}", table_cell_style),
        Paragraph(f"<b>{data['grid_results'][g]['lightgbm']['mean_cv_pr_auc']:.4f}</b>" if data['group_champions'][g]['model_name']=='lightgbm' else f"{data['grid_results'][g]['lightgbm']['mean_cv_pr_auc']:.4f}", table_cell_style),
        Paragraph(f"<b>{data['grid_results'][g]['logistic_regression']['mean_cv_pr_auc']:.4f}</b>" if data['group_champions'][g]['model_name']=='logistic_regression' else f"{data['grid_results'][g]['logistic_regression']['mean_cv_pr_auc']:.4f}", table_cell_style),
        Paragraph(f"{data['grid_results'][g]['random_forest']['mean_cv_pr_auc']:.4f}", table_cell_style),
        Paragraph(f"{data['grid_results'][g]['knn']['mean_cv_pr_auc']:.4f}", table_cell_style),
        Paragraph(f"<b>{data['group_champions'][g]['model_name'].replace('_', ' ').title()} ({data['group_champions'][g]['mean_pr_auc']:.4f})</b>", table_cell_left)
    ]
    grid_table_data.append(row)

t_grid = Table(grid_table_data, colWidths=[80, 45, 62, 58, 62, 65, 50, 114])
t_grid.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
]))
story.append(t_grid)
story.append(Spacer(1, 6))

# Heatmap Figure
story.append(Image(str(heatmap_path), width=536, height=272))
story.append(Spacer(1, 8))

story.append(PageBreak())

# Section 3: Statistical Ablation & Parsimony
story.append(Paragraph("3. Statistical Ablation, Effect Size & Clinical Parsimony", h1_style))
story.append(Paragraph(
    "To establish if reduced feature subsets can replace the 21-feature survey without diagnostic loss, paired Wilcoxon Signed-Rank tests, Cohen's $d_z$ effect sizes, and Parsimony Efficiency Indices were computed across identical 5-fold cross-validation splits.",
    body_style
))
story.append(Spacer(1, 4))

stat_headers = [
    Paragraph("<b>Feature Group</b>", table_header_style),
    Paragraph("<b>k</b>", table_header_style),
    Paragraph("<b>Champion Model</b>", table_header_style),
    Paragraph("<b>PR-AUC (±Std)</b>", table_header_style),
    Paragraph("<b>Diff vs Combined</b>", table_header_style),
    Paragraph("<b>Cohen's dz</b>", table_header_style),
    Paragraph("<b>Wilcoxon p</b>", table_header_style),
    Paragraph("<b>Parsimony</b>", table_header_style),
    Paragraph("<b>Statistical Equivalence</b>", table_header_style)
]
stat_table_data = [stat_headers]
for g in feature_groups:
    champ = data["group_champions"][g]
    stat = data["statistical_comparisons"][g]
    is_equiv = "Yes (p > 0.05)" if stat["wilcoxon_p_val"] > 0.05 and g != "combined_all" else ("Baseline" if g == "combined_all" else "No (p < 0.05)")
    row = [
        Paragraph(g.replace("_", " ").title(), table_cell_left),
        Paragraph(str(champ["num_features"]), table_cell_style),
        Paragraph(champ["model_name"].replace("_", " ").title(), table_cell_style),
        Paragraph(f"{champ['mean_pr_auc']:.4f} ±{champ['std']:.4f}", table_cell_style),
        Paragraph(f"{stat['mean_diff_vs_combined']:+.4f}", table_cell_style),
        Paragraph(f"{stat['cohens_dz']:.2f}", table_cell_style),
        Paragraph(f"{stat['wilcoxon_p_val']:.4f}", table_cell_style),
        Paragraph(f"<b>{stat['parsimony_efficiency_index']:.4f}</b>", table_cell_style),
        Paragraph(f"<b>{is_equiv}</b>" if "Yes" in is_equiv else is_equiv, table_cell_style)
    ]
    stat_table_data.append(row)

t_stat = Table(stat_table_data, colWidths=[75, 20, 75, 78, 62, 48, 55, 48, 75])
t_stat.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D9488")),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 2.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
]))
story.append(t_stat)
story.append(Spacer(1, 6))

story.append(Image(str(box_parsimony_path), width=536, height=214))
story.append(Spacer(1, 8))

# Section 4: Double Stacking Meta-Learners
story.append(Paragraph("4. Double Meta-Learner Stacking Architectures (Level-2 Stacking)", h1_style))
story.append(Paragraph(
    "Two stacking meta-learners (L2-Regularized Logistic Regression) were fitted on out-of-fold cross-validated probability matrices Z (202,946 rows by K columns):<br/>"
    "&bull; <b>Design A (Homogeneous Algorithm Diversity)</b>: Combines 5 distinct algorithms on <code>Combined All</code> (Z_A: 202,946 &times; 5).<br/>"
    "&bull; <b>Design B (Heterogeneous Domain Diversity)</b>: Combines 6 domain champions (Z_B: 202,946 &times; 6), yielding interpretable domain importance weights.",
    body_style
))
story.append(Spacer(1, 4))
story.append(Image(str(weights_path), width=536, height=185))
story.append(Spacer(1, 8))

story.append(PageBreak())

# Section 5: Holdout Test Set Evaluation
story.append(Paragraph("5. Final 20% Unseen Holdout Test Evaluation (N = 50,736)", h1_style))
story.append(Paragraph(
    "All champion architectures were refitted on the full 80% training set (N = 202,946) and scored against the completely untouched 20% holdout test partition (N = 50,736, positive prevalence 13.93%).",
    body_style
))
story.append(Spacer(1, 4))

holdout_headers = [
    Paragraph("<b>Candidate Architecture</b>", table_header_style),
    Paragraph("<b>Test PR-AUC</b>", table_header_style),
    Paragraph("<b>Test ROC-AUC</b>", table_header_style),
    Paragraph("<b>Optimal τ*</b>", table_header_style),
    Paragraph("<b>Calibrated F1</b>", table_header_style),
    Paragraph("<b>Recall (Sens)</b>", table_header_style),
    Paragraph("<b>Precision</b>", table_header_style),
    Paragraph("<b>Default F1 (0.50)</b>", table_header_style),
]
holdout_table_data = [holdout_headers]
for ev in eval_rows:
    row = [
        Paragraph(ev["name"], table_cell_left),
        Paragraph(f"<b>{ev['holdout_pr_auc']:.4f}</b>" if "Design A" in ev["name"] else f"{ev['holdout_pr_auc']:.4f}", table_cell_style),
        Paragraph(f"{ev['holdout_roc_auc']:.4f}", table_cell_style),
        Paragraph(f"{ev['tau_star']:.2f}", table_cell_style),
        Paragraph(f"<b>{ev['f1_calibrated']:.4f}</b>" if "Design A" in ev["name"] else f"{ev['f1_calibrated']:.4f}", table_cell_style),
        Paragraph(f"<b>{ev['recall_calibrated']*100:.1f}%</b>" if "Design B" in ev["name"] else f"{ev['recall_calibrated']*100:.1f}%", table_cell_style),
        Paragraph(f"{ev['precision_calibrated']*100:.1f}%", table_cell_style),
        Paragraph(f"{ev['f1_default']:.4f}", table_cell_style)
    ]
    holdout_table_data.append(row)

t_holdout = Table(holdout_table_data, colWidths=[150, 55, 55, 50, 58, 58, 55, 55])
t_holdout.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 3),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
]))
story.append(t_holdout)
story.append(Spacer(1, 6))

story.append(Image(str(cm_path), width=536, height=176))
story.append(Spacer(1, 8))

# Section 6: Scientific Discussion & Recommendations
story.append(Paragraph("6. Clinical Screening Recommendations & Synthesis", h1_style))
rec_text = """
<b>1. Deployment Recommendation for Primary Care Screening:</b> Deploy the <b>Hybrid RFE 10-Feature Battery</b> using <b>Design A Stacking</b> or <b>LightGBM</b>. It eliminates 11 burdensome questionnaire fields with zero statistically significant drop in diagnostic efficacy (PR-AUC 0.4191 vs 0.4212, <i>p</i> = 0.4375).
<br/><br/>
<b>2. Mandatory Decision Threshold Calibration:</b> Uncalibrated default thresholds (τ = 0.50) fail in population health screening by missing >80% of true diabetics. Operating at the calibrated threshold (τ* = 0.23) captures >61% of all diabetes cases while maximizing overall F1 utility.
<br/><br/>
<b>3. Multi-View Clinical Validation:</b> Meta-Learner Design B confirms that physiological biomarkers (HighBP, HighChol, BMI) and modifiable behaviors (Diet, Physical Activity, Smoking) contribute 85%+ of the composite log-odds risk signal, confirming lifestyle interventions as viable co-targets in diabetes prevention.
"""
story.append(Paragraph(rec_text, body_style))

# Build Document
doc.build(story, canvasmaker=NumberedCanvas)
print(f"Publication-grade PDF successfully compiled at: {PDF_PATH.resolve()}")
