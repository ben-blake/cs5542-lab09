"""
Visualization utility for automatic chart generation.

This module provides automatic chart selection based on DataFrame column types,
following simple heuristics to create appropriate visualizations for common
data patterns.
"""

import pandas as pd
import altair as alt
from typing import Optional


def _detect_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Categorize DataFrame columns into datetime, numeric, and categorical types.

    Args:
        df: pandas DataFrame to analyze

    Returns:
        Dictionary with keys 'datetime', 'numeric', 'categorical' mapping to
        lists of column names of each type

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=3),
        ...     'sales': [100, 200, 150],
        ...     'region': ['East', 'West', 'East']
        ... })
        >>> _detect_column_types(df)
        {'datetime': ['date'], 'numeric': ['sales'], 'categorical': ['region']}
    """
    column_types = {
        'datetime': [],
        'numeric': [],
        'categorical': []
    }

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types['datetime'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types['numeric'].append(col)
        else:
            # Non-numeric, non-datetime columns are categorical
            column_types['categorical'].append(col)

    return column_types


def auto_chart(df: pd.DataFrame) -> Optional[alt.Chart]:
    """
    Automatically select and create the best chart for the DataFrame.

    Uses heuristic rules based on column types:
    - 1 datetime + 1 numeric: Line chart (time series)
    - 1 categorical + 1 numeric: Bar chart (horizontal if >7 categories)
    - 2 numerics: Scatter plot
    - Otherwise: Returns None (data shown as table)

    Args:
        df: pandas DataFrame with query results

    Returns:
        Altair Chart object with interactive features, or None if no suitable
        visualization can be determined

    Examples:
        >>> # Time series data
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2024-01-01', periods=10),
        ...     'revenue': [100, 120, 115, 130, 125, 140, 135, 150, 145, 160]
        ... })
        >>> chart = auto_chart(df)  # Returns line chart

        >>> # Categorical data
        >>> df = pd.DataFrame({
        ...     'product': ['A', 'B', 'C', 'D'],
        ...     'sales': [100, 200, 150, 180]
        ... })
        >>> chart = auto_chart(df)  # Returns bar chart

        >>> # Numeric correlation
        >>> df = pd.DataFrame({
        ...     'price': [10, 20, 30, 40],
        ...     'quantity': [100, 80, 60, 40]
        ... })
        >>> chart = auto_chart(df)  # Returns scatter plot
    """
    # Edge case: empty or single-column DataFrame
    if df is None or df.empty or len(df.columns) < 2:
        return None

    # Cap rows to prevent browser lag — charts beyond 500 rows are unreadable anyway
    MAX_CHART_ROWS = 500
    if len(df) > MAX_CHART_ROWS:
        df = df.head(MAX_CHART_ROWS)

    # Detect column types
    col_types = _detect_column_types(df)

    # Rule 1: 1 datetime + 1 numeric → Line chart
    if len(col_types['datetime']) >= 1 and len(col_types['numeric']) >= 1:
        date_col = col_types['datetime'][0]  # Use first datetime column
        numeric_col = col_types['numeric'][0]  # Use first numeric column

        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X(date_col, title=date_col),
            y=alt.Y(numeric_col, title=numeric_col),
            tooltip=[date_col, numeric_col]
        ).properties(
            width=600,
            height=400,
            title=f"{numeric_col} over {date_col}"
        ).interactive()

        return chart

    # Rule 2: 1 categorical + 1 numeric → Bar chart
    if len(col_types['categorical']) >= 1 and len(col_types['numeric']) >= 1:
        # Prefer aggregate-looking numeric columns (counts, totals, averages, scores)
        _AGG_SUFFIXES = ('_count', '_total', '_sum', '_avg', '_revenue', '_score', '_days', '_time')
        agg_cols = [c for c in col_types['numeric'] if c.lower().endswith(_AGG_SUFFIXES)]
        numeric_col = agg_cols[0] if agg_cols else col_types['numeric'][-1]

        # Prefer non-ID categorical columns (skip columns ending in _id or _key)
        non_id_cats = [c for c in col_types['categorical'] if not c.lower().endswith(('_id', '_key', '_prefix'))]
        cat_col = non_id_cats[0] if non_id_cats else col_types['categorical'][0]

        # Count unique categories to decide orientation
        num_categories = df[cat_col].nunique()

        # Horizontal bar chart if more than 7 categories for readability
        if num_categories > 7:
            chart = alt.Chart(df).mark_bar().encode(
                y=alt.Y(cat_col, title=cat_col, sort='-x'),
                x=alt.X(numeric_col, title=numeric_col),
                tooltip=[cat_col, numeric_col]
            ).properties(
                width=600,
                height=400,
                title=f"{numeric_col} by {cat_col}"
            ).interactive()
        else:
            # Vertical bar chart for fewer categories
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(cat_col, title=cat_col),
                y=alt.Y(numeric_col, title=numeric_col),
                tooltip=[cat_col, numeric_col]
            ).properties(
                width=600,
                height=400,
                title=f"{numeric_col} by {cat_col}"
            ).interactive()

        return chart

    # Rule 3: 2 numerics → Scatter plot
    if len(col_types['numeric']) >= 2:
        x_col = col_types['numeric'][0]  # First numeric column
        y_col = col_types['numeric'][1]  # Second numeric column

        chart = alt.Chart(df).mark_point(filled=True, size=100).encode(
            x=alt.X(x_col, title=x_col),
            y=alt.Y(y_col, title=y_col),
            tooltip=[x_col, y_col]
        ).properties(
            width=600,
            height=400,
            title=f"{y_col} vs {x_col}"
        ).interactive()

        return chart

    # Otherwise: No suitable visualization found
    return None
