import matplotlib.pyplot as plt

datasets = ["PubMedQA", "MEDIQA2019", "MASH-QA", "LiveMedQA2017"]
    
# Model scores(put the scores here)
model_scores ={
    "Gpt-Medical": [],
    "Gpt-Medical_C": [],
    "Alpaca-Lora-7B": [],
    "Alpaca-Lora-7B_C": [],
    "MedAlpaca-7B": [],
    "MedAlpaca-7B_C": [],
}

# Colors for each model family
colors = {
    "Gpt-Medical": "blue",
    "Alpaca-Lora": "green",
    "MedAlpaca": "orange",
}

# Line styles
line_styles = {
    "solid": "-",
    "dashed": "--",
}

# Fallback color for missing model families
fallback_color = "gray"
fallback_marker = "x"

# Plot the F1 scores
plt.figure(figsize=(14, 8))

for model, scores in model_scores.items():
    if model.endswith("_C"):
        model_family = model.replace("-7B_C", "").replace("_C", "")
    else:
        model_family = model.replace("-7B", "")

    # Use the correct line style for the base or fine-tuned models
    line_style = line_styles["solid"] if "_C" not in model else line_styles["dashed"]
    label = model.replace("_", " ")  # Make the label more readable

    # Use the fallback color if the model family is not found in the color dictionary
    color = colors.get(model_family, fallback_color)
    # marker = markers.get(model_family, fallback_marker)

    # Plot the line
    plt.plot(
        datasets,
        scores,
        line_style,
        label=label,
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=8,
    )

# Add custom legend for solid and dashed lines
handles = []
for model_family, color in colors.items():
    handles.append(plt.Line2D([0], [0], color=color, linestyle="-", linewidth=2, label=f"{model_family}"))
    handles.append(plt.Line2D([0], [0], color=color, linestyle="--", linewidth=2, label=f"{model_family}_C"))

# Customize the plot
plt.title("... Scores Across Datasets", fontsize=18, fontweight="bold")
plt.xlabel("Dataset", fontsize=14)
plt.ylabel("... Score", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(handles=handles, fontsize=10, title="Models", title_fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()