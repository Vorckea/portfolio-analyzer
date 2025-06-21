def get_summary_card_html(title: str, subtitle: str, body_html: str) -> str:
    """Generate a styled HTML card for displaying summary results."""
    style = """
    <style>
        .summary-container { display: flex; justify-content: flex-start; padding: 1em 0; gap: 20px; }
        .summary-card {
            background-color: #fdfdfd; border: 1px solid #e8e8e8; border-radius: 10px;
            padding: 25px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); color: #333;
            width: 100%; max-width: 600px; flex: 1;
        }
        .summary-card h3 {
            margin-top: 0; margin-bottom: 15px; font-size: 1.3em; font-weight: 600;
            color: #1a1a1a; border-bottom: 1px solid #f0f0f0; padding-bottom: 15px;
        }
        .summary-card .subtitle { font-size: 0.95em; color: #555; margin-top: -15px; margin-bottom: 20px; }
        .summary-card h4 { margin-top: 25px; margin-bottom: 10px; font-size: 1.1em; font-weight: 600; color: #444; }

        /* Metric Grid Styles (for Optimization & Monte Carlo) */
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .metric { display: flex; flex-direction: column; text-align: center; background-color: #f9f9f9; padding: 12px; border-radius: 8px; }
        .metric-label { font-size: 0.85em; color: #666; margin-bottom: 5px; }
        .metric-value { font-size: 1.25em; font-weight: 600; color: #005a9e; }

        /* Weights List Styles (for Optimization) */
        .weights-list { list-style-type: none; padding-left: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 0 25px; }
        .weights-list li { display: flex; justify-content: space-between; padding: 9px 5px; border-bottom: 1px solid #f5f5f5; }
        .weights-list li:last-child { border-bottom: none; }
        .weights-list li:nth-last-child(2):not(:first-child) { border-bottom: none; } /* Handle even/odd cases for 2 columns */
        .ticker-name { font-weight: 500; }
        .ticker-weight { font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; font-weight: 600; }

        /* Table Styles (for Backtesting) */
        .summary-table { width: 100%; border-collapse: collapse; }
        .summary-table th, .summary-table td { text-align: left; padding: 12px 15px; border-bottom: 1px solid #f5f5f5; }
        .summary-table th { font-weight: 600; color: #444; background-color: #f9f9f9; }
        .summary-table td:nth-child(1) { font-weight: 500; }
        .summary-table tr:last-child td { border-bottom: none; }
        .summary-table td:not(:first-child) { text-align: right; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace; }
    </style>
    """  # noqa: E501
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    html = f"""
    {style}
    <div class="summary-container">
        <div class="summary-card">
            <h3>{title}</h3>
            {subtitle_html}
            {body_html}
        </div>
    </div>
    """
    return html
