"""
Jupyter Notebook Patterns - Study Script
This demonstrates what you'd do in an actual notebook.
In a real notebook, each function would be in separate cells.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def cell_1_load_and_explore():
    """Cell 1: Load and explore data"""
    print("=== Cell 1: Load & Explore ===\n")

    # Create sample data
    np.random.seed(42)
    data = np.random.randn(1000, 5)
    df = pd.DataFrame(data, columns=['feature_' + str(i) for i in range(5)])

    print(f"Shape: {df.shape}")
    print(f"\nFirst 3 rows:\n{df.head(3)}\n")
    print(f"Statistics:\n{df.describe()}")

    return df


def cell_2_timing_benchmark(df):
    """Cell 2: Benchmark operations"""
    print("\n=== Cell 2: Timing Benchmark ===\n")

    # Simulate %timeit - run many times
    iterations = 100
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = df.mean()
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000  # Convert to ms
    print(f"df.mean() averaged over {iterations} runs: {avg_time:.3f} ms")


def cell_3_visualization(df):
    """Cell 3: Create visualization"""
    print("\n=== Cell 3: Visualization ===\n")

    plt.figure(figsize=(10, 4))

    # Plot 1: Distribution
    plt.subplot(1, 2, 1)
    plt.hist(df['feature_0'], bins=30, edgecolor='black')
    plt.title('Distribution of feature_0')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plot 2: Correlation
    plt.subplot(1, 2, 2)
    corr = df.corr()
    plt.imshow(corr, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Feature Correlation')

    plt.tight_layout()
    plt.savefig('notebook_visualization.png', dpi=100)
    print("Visualization saved to notebook_visualization.png")
    print("In a notebook, plt.show() displays it inline in the cell below.")


def cell_4_model_summary():
    """Cell 4: Create a summary table"""
    print("\n=== Cell 4: Model Comparison ===\n")

    results = pd.DataFrame({
        'Model': ['Linear', 'Random Forest', 'XGBoost', 'Neural Net'],
        'Accuracy': [0.72, 0.85, 0.88, 0.91],
        'Training Time (s)': [0.1, 2.3, 5.1, 45.6],
        'Parameters': [102, 50_000, 25_000, 1_200_000]
    })

    print(results.to_string(index=False))
    print(f"\nBest accuracy: {results.loc[results['Accuracy'].idxmax(), 'Model']}")


def cell_5_cleanup():
    """Cell 5: Memory check"""
    print("\n=== Cell 5: Memory Check ===\n")
    print("In a real notebook, variables from previous cells are still in memory.")
    print("This is why 'Kernel > Restart & Run All' is important before sharing.")
    print("If memory leaks happen, use: del variable_name; gc.collect()")


if __name__ == "__main__":
    print("Notebook Patterns Demonstration\n")
    print("Each function represents a cell in a Jupyter notebook.\n")

    df = cell_1_load_and_explore()
    cell_2_timing_benchmark(df)
    cell_3_visualization(df)
    cell_4_model_summary()
    cell_5_cleanup()

    print("\n[OK] In a real notebook, you'd run each cell independently,")
    print("     see outputs inline, and be able to modify and re-run anytime.")
