import os
import statistics
import warnings
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from polars.functions import first

warnings.filterwarnings(action="ignore", category=pl.exceptions.PolarsInefficientMapWarning)  # pyright: ignore[reportArgumentType]


def load_data() -> pl.DataFrame:
    df = pl.read_csv("data.csv", schema_overrides={"number": pl.Utf8})

    df = df.with_columns(
        pl.col("number")
        .map_elements(lambda x: Decimal(x) if x != "inf" else float("inf"), return_dtype=pl.Object)
        .alias("number_decimal")
    )

    # Update the timestamp column to be datetime
    # All timestamps are on the 21/10/2025, +13:00 timezone
    # And look like just "14:23" in the CSV
    df = df.with_columns(
        (pl.lit("2025-10-21 ") + pl.col("timestamp") + pl.lit(":00+13:00"))
        .str.strptime(pl.Datetime(time_zone="Pacific/Auckland"), "%Y-%m-%d %H:%M:%S%z")
        .alias("timestamp_parsed")
    )

    return df


def print_troublesome_numbers(df: pl.DataFrame):
    print(df.select(["submitter", "number", "number_decimal"]))
    print("\nBrook's number (1.8e308):")
    brook_row = df.filter(pl.col("submitter") == "Brook")
    print(f"Original: {brook_row.select('number').item()}")
    print(f"Decimal: {brook_row.select('number_decimal').item()}")
    print(df.describe())
    print(df)


def draw_summary_plots(df: pl.DataFrame) -> None:
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    print("Creating plots...")

    # 1. Histogram of numbers (use string number column for reasonable values)
    # Filter for visualization (exclude very large numbers and inf)
    hist_data = df.filter(
        (pl.col("number") != "inf") & (pl.col("number").str.len_chars() < 15)  # Reasonable sized numbers
    )

    # Convert to float for plotting
    hist_values = hist_data.select(pl.col("number").cast(pl.Float64).alias("numeric_value")).to_dict(as_series=False)[
        "numeric_value"
    ]

    fig_hist = go.Figure(data=[go.Histogram(x=hist_values, nbinsx=15)])
    fig_hist.update_layout(
        title="Distribution of Numbers (Reasonable Range)", xaxis_title="Number Value", yaxis_title="Count"
    )
    fig_hist.write_html(Path("plots/histogram_numbers.html"))
    print("✓ Histogram saved")

    # 2. Box plot of numbers (same filtered data)
    fig_box = go.Figure(data=[go.Box(y=hist_values, name="Numbers")])
    fig_box.update_layout(title="Box Plot of Numbers (Reasonable Range)", yaxis_title="Number Value")
    fig_box.write_html(Path("plots/boxplot_numbers.html"))
    print("✓ Box plot saved")

    # 3. Pie chart of submitter count
    submitter_counts = df.group_by("submitter").len().sort("len", descending=True).to_dict(as_series=False)

    fig_pie_submitter = go.Figure(
        data=[go.Pie(labels=submitter_counts["submitter"], values=submitter_counts["len"], hole=0.3)]
    )
    fig_pie_submitter.update_layout(title="Submissions by Person")
    fig_pie_submitter.write_html(Path("plots/pie_submitters.html"))
    print("✓ Submitter pie chart saved")

    # 4. Pie chart of method count
    method_counts = df.group_by("method").len().sort("len", descending=True).to_dict(as_series=False)

    fig_pie_method = go.Figure(data=[go.Pie(labels=method_counts["method"], values=method_counts["len"], hole=0.3)])
    fig_pie_method.update_layout(title="Submissions by Method (Channel vs Thread vs Reply)")
    fig_pie_method.write_html(Path("plots/pie_methods.html"))
    print("✓ Method pie chart saved")

    # 5. Pie chart of type count (new column from updated CSV)
    type_counts = df.group_by("type").len().sort("len", descending=True).to_dict(as_series=False)

    fig_pie_type = go.Figure(data=[go.Pie(labels=type_counts["type"], values=type_counts["len"], hole=0.3)])
    fig_pie_type.update_layout(title="Submissions by Number Type")
    fig_pie_type.write_html(Path("plots/pie_types.html"))
    print("✓ Type pie chart saved")

    # 6. Scatter plot of timestamp_parsed vs number (filtered for reasonable numbers)
    scatter_data = (
        df.filter((pl.col("number") != "inf") & (pl.col("number").str.len_chars() < 15))
        .select(["timestamp_parsed", pl.col("number").cast(pl.Float64).alias("numeric_value"), "submitter", "type"])
        .to_dict(as_series=False)
    )

    # Calculate trend line (linear regression on log scale)
    # Convert timestamps to numeric values (seconds since first timestamp)
    timestamps: list[datetime] = scatter_data["timestamp_parsed"]
    first_time = min(timestamps)
    x_numeric = np.array([(t - first_time).total_seconds() for t in timestamps])
    y_values = np.array(scatter_data["numeric_value"])

    # Filter out zeros and negative values for log scale
    valid_mask = y_values > 0
    x_valid = x_numeric[valid_mask]
    y_valid = y_values[valid_mask]
    y_log = np.log10(y_valid)

    # Calculate linear regression on log scale
    correlation = np.corrcoef(x_valid, y_log)[0, 1]
    slope, intercept = np.polyfit(x_valid, y_log, 1)

    # Generate trend line points
    x_trend = np.linspace(x_numeric.min(), x_numeric.max(), 100)
    y_trend_log = slope * x_trend + intercept
    y_trend = 10**y_trend_log

    # Convert x_trend back to timestamps for plotting
    trend_timestamps = [first_time + timedelta(seconds=float(x)) for x in x_trend]

    # Create formula string
    formula = f"log₁₀(y) = {slope:.2e}·t + {intercept:.2f}"

    fig_scatter = go.Figure()

    # Add scatter points
    fig_scatter.add_trace(
        go.Scatter(
            x=scatter_data["timestamp_parsed"],
            y=scatter_data["numeric_value"],
            mode="markers",
            name="Data",
            text=scatter_data["submitter"],
            marker={"size": 10, "opacity": 0.7},
            hovertemplate="<b>%{text}</b><br>Time: %{x}<br>Number: %{y}<extra></extra>",
        )
    )

    # Add trend line
    fig_scatter.add_trace(
        go.Scatter(
            x=trend_timestamps,
            y=y_trend,
            mode="lines",
            name=f"Trend (r={correlation:.3f})",
            line={"color": "red", "width": 2, "dash": "dash"},
            hovertemplate="Trend line<br>%{y:.2f}<extra></extra>",
        )
    )

    fig_scatter.update_layout(
        title=f"Numbers Over Time<br><sub>{formula} | Correlation: r={correlation:.3f}</sub>",
        xaxis_title="Timestamp",
        yaxis_title="Number Value",
        yaxis_type="log",  # Log scale for better visualization
        showlegend=True,
    )
    fig_scatter.write_html(Path("plots/scatter_time_vs_number.html"))
    print("✓ Scatter plot saved")

    # 7. Bar chart of submission frequency distribution
    # Count how many players submitted 1 time, 2 times, etc.
    player_submission_counts = df.group_by("submitter").len().select("len").to_dict(as_series=False)["len"]

    # Count frequency of each submission count

    frequency_dist = Counter(player_submission_counts)

    submission_counts = list(frequency_dist.keys())
    player_counts = list(frequency_dist.values())

    fig_bar = go.Figure(data=[go.Bar(x=submission_counts, y=player_counts)])
    fig_bar.update_layout(
        title="Distribution of Player Submission Frequency",
        xaxis_title="Number of Submissions per Player",
        yaxis_title="Number of Players",
        xaxis={"type": "category"},
    )
    fig_bar.write_html(Path("plots/bar_submission_frequency.html"))
    print("✓ Submission frequency bar chart saved")

    print("\nAll plots saved to 'plots/' directory!")


def print_fun_facts(df: pl.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FUN FACTS AND STATISTICS")
    print("=" * 60 + "\n")

    # Filter for finite numbers (exclude infinity)
    finite_df = df.filter(pl.col("number") != "inf")
    finite_numbers = finite_df.select("number_decimal").to_series().to_list()

    # Work with Decimal values to preserve precision (including Brook's 1.8e308)
    finite_decimals = [x for x in finite_numbers if x != float("inf")]

    # Find min and max using Decimal values
    min_value = min(finite_decimals)
    max_value = max(finite_decimals)

    # Also keep float versions for operations that need them (filtering out inf values)
    finite_numeric = [float(x) for x in finite_decimals if float(x) != float("inf")]

    # Create a temporary column with float values for efficient filtering
    finite_df_with_float = finite_df.with_columns(
        pl.col("number_decimal").map_elements(lambda x: float(x), return_dtype=pl.Float64).alias("number_float")
    )

    # Find submitters for min and max
    min_submitter = (
        finite_df.filter(pl.col("number_decimal").map_elements(lambda x: x == min_value, return_dtype=pl.Boolean))
        .select("submitter")
        .item()
    )

    max_submitter = (
        finite_df.filter(pl.col("number_decimal").map_elements(lambda x: x == max_value, return_dtype=pl.Boolean))
        .select("submitter")
        .item()
    )

    # === BASIC STATISTICS ===
    print("📊 BASIC STATISTICS")
    print("-" * 40)
    print(f"Total submissions: {len(df)}")
    print(f"Smallest number: {min_value} (submitted by {min_submitter})")
    print(f"Largest finite number: {max_value} (submitted by {max_submitter})")

    # Calculate mean using Decimal values
    mean_val = sum(finite_decimals, Decimal(0)) / len(finite_decimals)
    print(f"Mean: {mean_val:.2e}")

    # Calculate median using Decimal values
    sorted_decimals = sorted(finite_decimals)
    median_val = (
        sorted_decimals[len(sorted_decimals) // 2]
        if len(sorted_decimals) % 2 == 1
        else (sorted_decimals[len(sorted_decimals) // 2 - 1] + sorted_decimals[len(sorted_decimals) // 2]) / 2
    )
    print(f"Median: {median_val}")

    if len(finite_decimals) > 1:
        stdev = statistics.stdev(finite_decimals)
        print(f"Standard deviation: {stdev:.2e}")

    # Calculate range using Decimal values
    range_val = max_value - min_value
    print(f"Range (max - min): {range_val:.2e}")

    # === NEGATIVE NUMBERS ===
    print(f"\n🔻 NEGATIVE NUMBERS")
    print("-" * 40)
    negative_count = sum(1 for x in finite_numeric if x < 0)
    print(f"Count: {negative_count}")
    if negative_count > 0:
        negative_nums = finite_df_with_float.filter(pl.col("number_float") < 0)
        for row in negative_nums.iter_rows(named=True):
            print(f"  • {row['submitter']}: {row['number_decimal']}")

    # === MEME NUMBERS ===
    print(f"\n😄 MEME NUMBERS")
    print("-" * 40)
    meme_count = len(df.filter(pl.col("type") == "Meme Number"))
    print(f"Total meme numbers: {meme_count}")
    meme_entries = df.filter(pl.col("type") == "Meme Number")
    for row in meme_entries.iter_rows(named=True):
        print(f"  • {row['submitter']}: {row['number']} (message: '{row['message']}')")

    # === MATHEMATICAL CONSTANTS ===
    print(f"\n🔢 MATHEMATICAL CONSTANTS")
    print("-" * 40)
    symbol_entries = df.filter(pl.col("type") == "Symbol")
    print(f"Count: {len(symbol_entries)}")
    for row in symbol_entries.iter_rows(named=True):
        print(f"  • {row['submitter']}: {row['message']} → {row['number_decimal']}")

    # === BULLSHIT ENTRIES ===
    print(f"\n💩 'BULLSHIT' ENTRIES")
    print("-" * 40)
    bs_entries = df.filter(pl.col("type") == "Bullshit")
    print(f"Count: {len(bs_entries)}")
    for row in bs_entries.iter_rows(named=True):
        comment = f" - {row['comment']}" if row["comment"] else ""
        print(f"  • {row['submitter']}: {row['message']}{comment}")

    # === INTEGER VS FLOAT ===
    print(f"\n🔢 INTEGER vs FLOAT")
    print("-" * 40)
    integer_count = sum(1 for x in finite_numeric if x == int(x))
    float_count = len(finite_numeric) - integer_count
    print(f"Integers: {integer_count} ({integer_count / len(finite_numeric) * 100:.1f}%)")
    print(f"Floats: {float_count} ({float_count / len(finite_numeric) * 100:.1f}%)")

    # === MAGNITUDE ANALYSIS ===
    print(f"\n📏 MAGNITUDE ANALYSIS")
    print("-" * 40)
    gt_100 = sum(1 for x in finite_numeric if x > 100)
    gt_1000 = sum(1 for x in finite_numeric if x > 1000)
    print(f"Numbers > 100: {gt_100} ({gt_100 / len(finite_numeric) * 100:.1f}%)")
    print(f"Numbers > 1000: {gt_1000} ({gt_1000 / len(finite_numeric) * 100:.1f}%)")
    if max(finite_numeric) > 0 and min(finite_numeric) != 0:
        ratio = abs(max(finite_numeric) / min(finite_numeric))
        print(f"Ratio of largest to smallest: {ratio:.2e}")

    # === DUPLICATE NUMBERS ===
    print(f"\n🔁 DUPLICATE NUMBERS")
    print("-" * 40)
    # Use the existing finite_df_with_float that already has number_float column
    duplicates = (
        finite_df_with_float.group_by("number_float")
        .agg([pl.col("submitter").alias("submitters"), pl.len().alias("count")])
        .filter(pl.col("count") > 1)
    )

    if len(duplicates) > 0:
        print(f"Count of duplicate values: {len(duplicates)}")
        for row in duplicates.iter_rows(named=True):
            submitters = ", ".join(row["submitters"])
            print(f"  • {row['number_float']}: submitted {row['count']} times by {submitters}")
    else:
        print("No duplicate numbers!")

    # === REPEAT SUBMITTERS ===
    print(f"\n👥 REPEAT SUBMITTERS")
    print("-" * 40)
    repeat_submitters = df.group_by("submitter").len().filter(pl.col("len") > 1).sort("len", descending=True)
    if len(repeat_submitters) > 0:
        print(f"Players who submitted multiple times: {len(repeat_submitters)}")
        for row in repeat_submitters.iter_rows(named=True):
            print(f"  • {row['submitter']}: {row['len']} submissions")
    else:
        print("Everyone submitted exactly once!")

    # === SUBMISSION METHOD ===
    print(f"\n💬 SUBMISSION METHOD")
    print("-" * 40)
    method_counts = df.group_by("method").len().sort("len", descending=True)
    for row in method_counts.iter_rows(named=True):
        print(f"  • {row['method']}: {row['len']} ({row['len'] / len(df) * 100:.1f}%)")

    # === TIME ANALYSIS ===
    print(f"\n⏰ TIME ANALYSIS")
    print("-" * 40)
    timestamps = df.select("timestamp_parsed").to_series().to_list()
    time_span = max(timestamps) - min(timestamps)
    print(f"First submission: {min(timestamps).strftime('%H:%M:%S')}")
    print(f"Last submission: {max(timestamps).strftime('%H:%M:%S')}")
    print(f"Time span: {time_span}")

    # === SPECIAL MENTIONS ===
    print(f"\n⭐ SPECIAL MENTIONS")
    print("-" * 40)

    # Leading zero
    leading_zero = df.filter(pl.col("message").str.contains("leading 0"))
    if len(leading_zero) > 0:
        submitter = leading_zero.select("submitter").to_series()[0]
        message = leading_zero.select("message").to_series()[0]
        print(f"  • Leading zero enthusiast: {submitter} with '{message}'")

    # Scientific notation - check message field since number is expanded
    sci_notation = df.filter(pl.col("message").str.contains("(?i)e[0-9]"))
    if len(sci_notation) > 0:
        submitter = sci_notation.select("submitter").to_series()[0]
        message = sci_notation.select("message").to_series()[0]
        print(f"  • Scientific notation user: {submitter} with {message}")

    # Infinity
    inf_entries = df.filter(pl.col("number") == "inf")
    if len(inf_entries) > 0:
        submitter = inf_entries.select("submitter").to_series()[0]
        message = inf_entries.select("message").to_series()[0]
        print(f"  • Infinity lover: {submitter} with {message}")

    print("\n" + "=" * 60)


def first_submission_from_each_submitter(df: pl.DataFrame) -> pl.DataFrame:
    # Sort by timestamp_parsed to ensure earliest submissions come first
    sorted_df = df.sort(["submitter", "timestamp_parsed"])

    # Get the first submission from each submitter
    first_submissions = sorted_df.unique(subset=["submitter"], keep="first")

    return first_submissions


def winner(first_submissions_df: pl.DataFrame) -> pl.DataFrame:
    # Pick a random winning row
    winning_row = first_submissions_df.sample(n=1, with_replacement=False, seed=123)
    return winning_row


def main():
    df = load_data()
    print_troublesome_numbers(df)
    draw_summary_plots(df)
    print_fun_facts(df)

    first_submissions_df = first_submission_from_each_submitter(df)
    print("\nFirst submission from each submitter:")
    for row in first_submissions_df.iter_rows(named=True):
        print(f"  • {row['submitter']}: {row['number']} at {row['timestamp_parsed']}")

    winning_row = winner(first_submissions_df)
    print("\n" + "=" * 60)
    print("🏆 RANDOMLY CHOSEN WINNER 🏆")
    for row in winning_row.iter_rows(named=True):
        print(f"Winner: {row['submitter']} with number {row['number']} submitted at {row['timestamp_parsed']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
