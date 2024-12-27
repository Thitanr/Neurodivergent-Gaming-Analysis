
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Generation (Synthetic Data)
def generate_dataset():
    """Generate a dataset simulating the effects of video games on neurodivergent individuals."""
    import random

    # Simulated data for neurodivergent individuals and their gaming habits
    data = {
        "Person": [f"Person_{i}" for i in range(1, 101)],
        "Neurodivergence_Type": random.choices(
            ["ADHD", "Autism", "Dyslexia", "High Intellectual Abilities"], k=100
        ),
        "Preferred_Game_Type": random.choices(
            ["Puzzle", "Adventure", "Strategy", "Action", "Simulation"], k=100
        ),
        "Average_Playtime_per_Week": [random.randint(1, 20) for _ in range(100)],
        "Concentration_Score": [random.uniform(1, 10) for _ in range(100)],
        "Calmness_Score": [random.uniform(1, 10) for _ in range(100)],
        "Problem_Solving_Score": [random.uniform(1, 10) for _ in range(100)],
    }

    return pd.DataFrame(data)

# Data Visualization and Analysis
def visualize_data(df):
    """Visualize the relationships and distributions in the dataset."""
    sns.set(style="whitegrid")

    # Distribution of Concentration, Calmness, and Problem Solving Scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[["Concentration_Score", "Calmness_Score", "Problem_Solving_Score"]])
    plt.title("Distribution of Scores")
    plt.ylabel("Scores")
    plt.xticks([0, 1, 2], ["Concentration", "Calmness", "Problem Solving"])
    plt.show()

    # Preferred Game Type Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Preferred_Game_Type", data=df, palette="muted")
    plt.title("Preferred Game Types by Neurodivergent Individuals")
    plt.xlabel("Game Type")
    plt.ylabel("Count")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[["Average_Playtime_per_Week", "Concentration_Score", "Calmness_Score", "Problem_Solving_Score"]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# Insights Extraction
def extract_insights(df):
    """Analyze and extract key insights from the dataset."""
    insights = {}

    # Average scores by Neurodivergence Type
    scores_by_type = df.groupby("Neurodivergence_Type")[
        ["Concentration_Score", "Calmness_Score", "Problem_Solving_Score"]
    ].mean()

    insights["average_scores_by_type"] = scores_by_type

    # Game Type Preferences
    game_type_preferences = df["Preferred_Game_Type"].value_counts()
    insights["game_type_preferences"] = game_type_preferences

    return insights

# Main Function
def main():
    """Main function to execute the analysis pipeline."""
    print("Generating dataset...")
    df = generate_dataset()
    print("Dataset generated successfully!\n")
    print(df.head())

    print("\nVisualizing data...")
    visualize_data(df)

    print("\nExtracting insights...")
    insights = extract_insights(df)
    print("Insights:")
    print(insights["average_scores_by_type"])
    print("\nGame Type Preferences:")
    print(insights["game_type_preferences"])

if __name__ == "__main__":
    main()
