#!/bin/bash

# Default values
binary=""
declare -a n_points
declare -a n_dims
n_jobs=24

mkdir -p ./tmp_benchmark

# Function to parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary=*)
            binary="${1#*=}"
            ;;
        --n_points=*)
            IFS=',' read -ra n_points <<< "${1#*=}"
            ;;
        --n_dims=*)
            IFS=',' read -ra n_dims <<< "${1#*=}"
            ;;
        --n_jobs=*)
            n_jobs="${1#*=}"
            ;;
    esac
    shift
done

# Check if binary path is provided
if [ -z "$binary" ]; then
    echo "Path to binary is required."
    exit 1
fi


cat > ./tmp_benchmark/gen.py <<EOF
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import sys

n_points = int(sys.argv[1])
n_features = int(sys.argv[2])

data, cluster_labels = make_blobs(n_samples=n_points, centers=20, center_box=(-150.0, 150.0), cluster_std=3, n_features=n_features)
df = pd.DataFrame(data.astype('float32')).sample(frac=1).reset_index(drop=True)

df.to_csv(sys.argv[3], header=False, index=False)
EOF

cat > ./tmp_benchmark/py_version.py <<EOF
import numpy as np
from sklearn.cluster import DBSCAN
import sklearn
import time
import sys
import pandas as pd

file_path = sys.argv[1]
points = np.loadtxt(file_path, delimiter=',')

eps = float(sys.argv[2])
min_samples = 5

start_time = time.time()
labels = DBSCAN(eps=10, min_samples=5, algorithm='kd_tree', n_jobs=int(sys.argv[3]), metric='euclidean').fit_predict(points)
end_time = time.time()

# Time taken for DBSCAN
time_taken = end_time - start_time

df = pd.DataFrame(labels.astype('int32')).reset_index(drop=True)
df.to_csv(sys.argv[4], header=False, index=False)

print(time_taken * 1000, "ms")
EOF

cat > ./tmp_benchmark/plot.py <<EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('results.csv')
dims = data['dims'].unique()

def annotate_bar(ax, bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

num_dims = len(dims)
fig, axs = plt.subplots(num_dims, 2, figsize=(20, 5 * num_dims))

for i, dim in enumerate(dims):
    subset = data[data['dims'] == dim]
    x = np.arange(len(subset))

    # Plotting elapsed time
    bars_cpp = axs[i, 0].bar(x - 0.2, subset['cpp_elapsed_time'], 0.4, label='C++ Elapsed Time (ms)', color='blue')
    bars_py = axs[i, 0].bar(x + 0.2, subset['py_elapsed_time'], 0.4, label='Python Elapsed Time (ms)', color='orange')

    # Annotate bars with the absolute values
    annotate_bar(axs[i, 0], bars_cpp, subset['cpp_elapsed_time'])
    annotate_bar(axs[i, 0], bars_py, subset['py_elapsed_time'])

    # Adjust y-axis limit to leave space for annotations
    max_height = max(subset['cpp_elapsed_time'].max(), subset['py_elapsed_time'].max())
    axs[i, 0].set_ylim(0, max_height * 1.2)  # Increase y-axis limit

    # Similar adjustments for memory usage plot
    bars_cpp = axs[i, 1].bar(x - 0.2, subset['cpp_mem_usage'], 0.4, label='C++ Memory Usage (kbytes)', color='blue')
    bars_py = axs[i, 1].bar(x + 0.2, subset['py_mem_usage'], 0.4, label='Python Memory Usage (kbytes)', color='orange')

    annotate_bar(axs[i, 1], bars_cpp, subset['cpp_mem_usage'])
    annotate_bar(axs[i, 1], bars_py, subset['py_mem_usage'])

    max_height = max(subset['cpp_mem_usage'].max(), subset['py_mem_usage'].max())
    axs[i, 1].set_ylim(0, max_height * 1.2)

    axs[i, 0].set_title(f'Elapsed Time Comparison for {dim} Dimensions')
    axs[i, 0].set_xticks(x)
    axs[i, 0].set_xticklabels(subset['points'])
    axs[i, 0].set_xlabel('Number of Points')
    axs[i, 0].set_ylabel('Elapsed Time (ms)')
    axs[i, 0].legend()

    axs[i, 1].set_title(f'Memory Usage Comparison for {dim} Dimensions')
    axs[i, 1].set_xticks(x)
    axs[i, 1].set_xticklabels(subset['points'])
    axs[i, 1].set_xlabel('Number of Points')
    axs[i, 1].set_ylabel('Memory Usage (kbytes)')
    axs[i, 1].legend()

plt.subplots_adjust(hspace=0.8)
plt.tight_layout()
plt.savefig('results.png', dpi=300)
EOF

echo "points,dims,cpp_elapsed_time,cpp_mem_usage,py_elapsed_time,py_mem_usage" > ./tmp_benchmark/results.csv

# Iterate over all permutations of n_points and n_dims
for points in "${n_points[@]}"; do
    for dims in "${n_dims[@]}"; do
        # Generate parameters file
        csv_file="./tmp_benchmark/params_${points}_${dims}.csv"

        python3 ./tmp_benchmark/gen.py $points $dims "$csv_file"

        echo "Computing for points: $points, dims: $dims..."

        eps=10

        # Run the C++ binary and capture the elapsed time and memory usage
        /usr/bin/time -v $binary "$csv_file" "$points" "$dims" $eps 5 $n_jobs ./tmp_benchmark/cpp_labels.txt &> ./tmp_benchmark/cpp_time_output.txt
        cpp_elapsed_time=$(grep -oP 'Elapsed = \d+μs or \K\d+(?=ms)' ./tmp_benchmark/cpp_time_output.txt || echo "OOM")
        cpp_mem_usage=$(grep -oP '(?<=Maximum resident set size \(kbytes\): )\d+' ./tmp_benchmark/cpp_time_output.txt)

        # Run the Python script and capture the elapsed time and memory usage
        /usr/bin/time -v python ./tmp_benchmark/py_version.py "$csv_file" $eps $n_jobs ./tmp_benchmark/py_labels.txt &> ./tmp_benchmark/py_time_output.txt
        py_elapsed_time=$(grep -oP '\d+\.\d+(?= ms)' ./tmp_benchmark/py_time_output.txt || echo "OOM")
        py_mem_usage=$(grep -oP '(?<=Maximum resident set size \(kbytes\): )\d+' ./tmp_benchmark/py_time_output.txt)

        echo "C++ Elapsed Time: $cpp_elapsed_time, Memory Usage: $cpp_mem_usage"
        echo "Python Elapsed Time: $py_elapsed_time, Memory Usage: $py_mem_usage"

        # Check if results are equal
        results_equal=$(cmp -s ./tmp_benchmark/cpp_labels.txt ./tmp_benchmark/py_labels.txt && echo true || echo false)
        echo "Results are equal: $results_equal"

        # Append the results to the CSV file
        echo "$points,$dims,$cpp_elapsed_time,$cpp_mem_usage,$py_elapsed_time,$py_mem_usage" >> ./tmp_benchmark/results.csv
    done
done

cd tmp_benchmark
python3 plot.py
