import dotenv

dotenv.load_dotenv()

import json
import os
import uuid
import pandas as pd
from typing import Dict, List, Tuple, Union


def export_html(
    prefix: str,
    results_dfs: List[Tuple[str, pd.DataFrame]],
    comparison_df: pd.DataFrame,
    output_dir: str = "outputs",
    allowlist_metrics: List[str] = ["LabeledScoreString(GPT4o-mini)"],
):
    os.makedirs(output_dir, exist_ok=True)
    html_file = f"{output_dir}/{prefix}-{str(uuid.uuid4())}.html"

    # Create a flattened version of the comparison DataFrame with separate columns for each statistic
    flat_data = []
    for model_name in comparison_df.index:
        row_data = {"Model": model_name}

        for metric, stats in comparison_df.loc[model_name].items():
            if allowlist_metrics and metric not in allowlist_metrics:
                continue
            for stat_name, stat_value in stats.items():
                # Create column headers like "accuracy_mean", "accuracy_std", etc.
                col_name = f"{metric}_{stat_name}"
                row_data[col_name] = stat_value

        flat_data.append(row_data)

    # Convert the flattened data to a DataFrame
    formatted_comparison = pd.DataFrame(flat_data).set_index("Model")

    # Style the metrics DataFrame
    styled_comparison_df = (
        formatted_comparison.style.set_properties(**{"text-align": "left", "padding": "8px", "border": "1px solid #ddd"})
        .format("{:.4f}")
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#42647b"),
                        ("color", "white"),
                        ("text-align", "left"),
                        ("padding", "4px"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
    )

    # Filter and style each results DataFrame
    styled_results_dfs: List[Tuple[str, pd.DataFrame]] = []
    for i, (prefix, df) in enumerate(results_dfs):
        # Define standard columns to always keep
        standard_cols = ["Question", "Ground_Truth", "Answer", "IsRag"]

        # Create a new DataFrame with only the columns we want to keep
        filtered_cols = []
        for col in df.columns:
            # Keep column if it's a standard column or in allowlist_metrics
            if col in standard_cols or (not allowlist_metrics) or (col in allowlist_metrics):
                filtered_cols.append(col)

        # Filter DataFrame to only include these columns
        filtered_df = df[filtered_cols]

        styled_df = filtered_df.style.set_properties(
            **{"text-align": "left", "padding": "8px", "border": "1px solid #ddd"}
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#6A5A2B"),
                        ("color", "white"),
                        ("text-align", "left"),
                        ("padding", "4px"),
                        ("font-weight", "bold"),
                    ],
                }
            ]
        )
        styled_results_dfs.append((prefix, styled_df))

    # Create HTML with multiple tables
    with open(html_file, "w") as f:
        f.write(
            f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Results</title>
            <style>
                body {{
                    font-family: monospace;
                    margin: 10px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #333;
                    border-bottom: 2px solid #6A5A2B;
                    padding-bottom: 4px;
                }}
                h2 {{
                    color: #555;
                    margin-top: 5px;
                }}
                .container {{
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    margin-bottom: 10px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 10px;
                }}
                caption {{
                    font-weight: bold;
                    font-size: 1.1em;
                    margin-bottom: 10px;
                    text-align: left;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                footer {{
                    margin-top: 30px;
                    color: #777;
                    font-size: 0.9em;
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <h1>Evaluation Results</h1>
            
            <div class="container">
                <h2>Summary</h2>
                {styled_comparison_df.to_html() if not formatted_comparison.empty else "<p>No summary metrics available</p>"}
            </div>
        """
        )

        # Add each results DataFrame
        for i, (df_name, styled_df) in enumerate(styled_results_dfs):
            f.write(
                f"""
            <div class="container">
                <h2>Results for {df_name}</h2>
                {styled_df.to_html()}
            </div>
            """
            )

        # Close the HTML
        f.write(
            f"""
            <footer>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </footer>
        </body>
        </html>
        """
        )

    return html_file


def export_csv(
    prefix: str,
    results: pd.DataFrame,
    output_dir: str = "outputs",
):
    os.makedirs(output_dir, exist_ok=True)
    csv_file = f"{output_dir}/{prefix}-{str(uuid.uuid4())}-comparison.csv"

    # Save to CSV
    results.to_csv(csv_file, index=False)

    return csv_file
