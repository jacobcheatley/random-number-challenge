import os
from decimal import Decimal

import plotly.express as px
import plotly.graph_objects as go
import polars as pl


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
    fig_hist.write_html("plots/histogram_numbers.html")
    print("✓ Histogram saved")

    # 2. Box plot of numbers (same filtered data)
    fig_box = go.Figure(data=[go.Box(y=hist_values, name="Numbers")])
    fig_box.update_layout(title="Box Plot of Numbers (Reasonable Range)", yaxis_title="Number Value")
    fig_box.write_html("plots/boxplot_numbers.html")
    print("✓ Box plot saved")

    # 3. Pie chart of submitter count
    submitter_counts = df.group_by("submitter").len().sort("len", descending=True).to_dict(as_series=False)

    fig_pie_submitter = go.Figure(
        data=[go.Pie(labels=submitter_counts["submitter"], values=submitter_counts["len"], hole=0.3)]
    )
    fig_pie_submitter.update_layout(title="Submissions by Person")
    fig_pie_submitter.write_html("plots/pie_submitters.html")
    print("✓ Submitter pie chart saved")

    # 4. Pie chart of method count
    method_counts = df.group_by("method").len().sort("len", descending=True).to_dict(as_series=False)

    fig_pie_method = go.Figure(data=[go.Pie(labels=method_counts["method"], values=method_counts["len"], hole=0.3)])
    fig_pie_method.update_layout(title="Submissions by Method (Channel vs Thread vs Reply)")
    fig_pie_method.write_html("plots/pie_methods.html")
    print("✓ Method pie chart saved")

    # 5. Pie chart of type count (new column from updated CSV)
    type_counts = df.group_by("type").len().sort("len", descending=True).to_dict(as_series=False)

    fig_pie_type = go.Figure(data=[go.Pie(labels=type_counts["type"], values=type_counts["len"], hole=0.3)])
    fig_pie_type.update_layout(title="Submissions by Number Type")
    fig_pie_type.write_html("plots/pie_types.html")
    print("✓ Type pie chart saved")

    # 6. Scatter plot of timestamp_parsed vs number (filtered for reasonable numbers)
    scatter_data = (
        df.filter((pl.col("number") != "inf") & (pl.col("number").str.len_chars() < 15))
        .select(["timestamp_parsed", pl.col("number").cast(pl.Float64).alias("numeric_value"), "submitter", "type"])
        .to_dict(as_series=False)
    )

    fig_scatter = go.Figure(
        data=go.Scatter(
            x=scatter_data["timestamp_parsed"],
            y=scatter_data["numeric_value"],
            mode="markers",
            text=scatter_data["submitter"],
            marker=dict(size=10, opacity=0.7),
            hovertemplate="<b>%{text}</b><br>Time: %{x}<br>Number: %{y}<extra></extra>",
        )
    )
    fig_scatter.update_layout(
        title="Numbers Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Number Value",
        yaxis_type="log",  # Log scale for better visualization
    )
    fig_scatter.write_html("plots/scatter_time_vs_number.html")
    print("✓ Scatter plot saved")

    # 7. Bar chart of submission frequency distribution
    # Count how many players submitted 1 time, 2 times, etc.
    player_submission_counts = df.group_by("submitter").len().select("len").to_dict(as_series=False)["len"]

    # Count frequency of each submission count
    from collections import Counter

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
    fig_bar.write_html("plots/bar_submission_frequency.html")
    print("✓ Submission frequency bar chart saved")

    print("\nAll plots saved to 'plots/' directory!")


def print_fun_facts(df: pl.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("FUN FACTS AND STATISTICS")
    print("=" * 60 + "\n")

    # Filter for finite numbers (exclude infinity)
    finite_df = df.filter(pl.col("number") != "inf")
    finite_numbers = finite_df.select("number_decimal").to_series().to_list()
    finite_numeric = [float(x) for x in finite_numbers]

    # === BASIC STATISTICS ===
    print("📊 BASIC STATISTICS")
    print("-" * 40)
    print(f"Total submissions: {len(df)}")
    print(
        f"Smallest number: {min(finite_numeric)} (submitted by {finite_df.filter(pl.col('number_decimal') == min(finite_numbers)).select('submitter').item()})"
    )
    print(
        f"Largest finite number: {max(finite_numeric)} (submitted by {finite_df.filter(pl.col('number_decimal') == max(finite_numbers)).select('submitter').item()})"
    )
    print(f"Mean: {sum(finite_numeric) / len(finite_numeric):.2f}")
    print(f"Median: {sorted(finite_numeric)[len(finite_numeric) // 2]:.2f}")

    import statistics

    if len(finite_numeric) > 1:
        print(f"Standard deviation: {statistics.stdev(finite_numeric):.2e}")
    print(f"Range (max - min): {max(finite_numeric) - min(finite_numeric):.2e}")

    # === NEGATIVE NUMBERS ===
    print(f"\n🔻 NEGATIVE NUMBERS")
    print("-" * 40)
    negative_count = sum(1 for x in finite_numeric if x < 0)
    print(f"Count: {negative_count}")
    if negative_count > 0:
        negative_nums = finite_df.filter(pl.col("number_decimal") < 0)
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
    duplicates = finite_df.group_by("number_decimal").len().filter(pl.col("len") > 1)
    if len(duplicates) > 0:
        print(f"Count of duplicate values: {len(duplicates)}")
        for row in duplicates.iter_rows(named=True):
            dup_entries = finite_df.filter(pl.col("number_decimal") == row["number_decimal"])
            submitters = ", ".join(dup_entries.select("submitter").to_series().to_list())
            print(f"  • {row['number_decimal']}: submitted {row['len']} times by {submitters}")
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
        print(
            f"  • Leading zero enthusiast: {leading_zero.select('submitter').item()} with '{leading_zero.select('message').item()}'"
        )

    # Scientific notation
    sci_notation = df.filter(pl.col("number").str.contains("e"))
    if len(sci_notation) > 0:
        print(
            f"  • Scientific notation user: {sci_notation.select('submitter').item()} with {sci_notation.select('number').item()}"
        )

    # Infinity
    inf_entries = df.filter(pl.col("number") == "inf")
    if len(inf_entries) > 0:
        print(
            f"  • Infinity lover: {inf_entries.select('submitter').item()} with {inf_entries.select('message').item()}"
        )

    print("\n" + "=" * 60)


def main():
    df = load_data()
    print_troublesome_numbers(df)
    draw_summary_plots(df)
    print_fun_facts(df)


if __name__ == "__main__":
    main()
