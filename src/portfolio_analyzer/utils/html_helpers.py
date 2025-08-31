def get_summary_card_html(title: str, subtitle: str, body_html: str) -> str:
    """Generate a styled HTML card for displaying summary results."""
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div class="summary-container">
        <div class="summary-card">
            <h3>{title}</h3>
            {subtitle_html}
            {body_html}
        </div>
    </div>
    """
    return html
