import dotenv

dotenv.load_dotenv()

import json
import os
import uuid
import pandas as pd
from typing import List, Tuple, Union


def export_html(
    prefix: str,
    results_dfs: List[Tuple[str, pd.DataFrame]],
    comparison_df: pd.DataFrame,
    output_dir: str = "outputs",
):
    """
    Export multiple result DataFrames and a metrics DataFrame to a single HTML file.

    Parameters:
    -----------
    name : str
        Base name for the output file
    results_dfs : List[Union[str, pd.DataFrame]]
        List of DataFrames containing detailed results
    comparison_df : pd.DataFrame
        DataFrame containing summary metrics
    output_dir : str, optional
        Directory to save the output file (default: "outputs")

    Returns:
    --------
    str
        Path to the generated HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    html_file = f"{output_dir}/{prefix}-{str(uuid.uuid4())}.html"

    # Style the metrics DataFrame
    styled_comparison_df = comparison_df.style.set_properties(
        **{"text-align": "left", "padding": "8px", "border": "1px solid #ddd"}
    ).set_table_styles(
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

    # Style each results DataFrame
    styled_results_dfs: List[Tuple[str, pd.DataFrame]] = []
    for i, (prefix, df) in enumerate(results_dfs):
        styled_df = df.style.set_properties(**{"text-align": "left", "padding": "8px", "border": "1px solid #ddd"}).set_table_styles(
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
                {styled_comparison_df.to_html() if not comparison_df.empty else "<p>No summary metrics available</p>"}
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
