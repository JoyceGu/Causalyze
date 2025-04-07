# Causalyze - Causal Analysis Data Platform

Causalyze is a web application focused on causal inference analysis, helping users discover causal relationships in their data.

## Features

- **Data Upload**: Support for Excel and CSV file uploads
- **Data Overview**: Automatic analysis and display of basic statistical information
- **Data Visualization**: Generate timeline charts for each feature to help understand data trends
- **Causal Inference**: Use advanced causal inference algorithms (EconML) to analyze how multiple features affect the target outcome
- **Result Interpretation**: Visually show which factors have substantial impact on the results

## How to Use

1. Upload a data file (Excel or CSV format)
2. View data overview in the left panel
3. Explore time trend visualizations for each feature in the center panel
4. Configure and run causal inference analysis in the right panel
5. View and interpret the analysis results

## Technology Stack

- Frontend: Streamlit
- Data Processing: Pandas, NumPy
- Visualization: Plotly
- Causal Inference: EconML

## Installation and Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Data Requirements

The uploaded data file should contain a time column, multiple feature columns, and at least one target outcome column. 