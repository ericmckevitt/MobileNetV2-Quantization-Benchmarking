from trtexec_results import all_runs
import pandas as pd
import matplotlib.pyplot as plt

# Flatten data for easy use
df = pd.json_normalize(all_runs, sep='_')

# Set style
plt.style.use('ggplot')

# Throughput Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['throughput_qps'])
plt.ylabel('Throughput (qps)')
plt.title('Throughput Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/throughput_comparison.png')
plt.close()

# Mean Latency Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['latency_mean_ms'])
plt.ylabel('Mean Latency (ms)')
plt.title('Mean Latency Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('img/mean_latency_comparison.png')
plt.close()

# GPU Compute Time Mean Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['gpu_compute_time_mean_ms'])
plt.ylabel('GPU Compute Mean Time (ms)')
plt.title('GPU Compute Time Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('img/gpu_compute_time_comparison.png')
plt.close()

# Enqueue Time Mean Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['enqueue_time_mean_ms'])
plt.ylabel('Enqueue Mean Time (ms)')
plt.title('Enqueue Time Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('img/enqueue_time_comparison.png')
plt.close()

# H2D Latency Mean Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['h2d_latency_mean_ms'])
plt.ylabel('H2D Mean Latency (ms)')
plt.title('Host-to-Device Latency Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('img/h2d_latency_comparison.png')
plt.close()

# D2H Latency Mean Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['d2h_latency_mean_ms'])
plt.ylabel('D2H Mean Latency (ms)')
plt.title('Device-to-Host Latency Comparison')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('img/d2h_latency_comparison.png')
plt.close()

# Total GPU Compute Time Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['total_gpu_compute_time_s'])
plt.ylabel('Total GPU Compute Time (s)')
plt.title('Total GPU Compute Time')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('img/total_gpu_compute_time.png')
plt.close()

# Total Host Walltime Comparison
plt.figure(figsize=(8,5))
plt.bar(df['device'] + ' ' + df['precision'], df['total_host_walltime_s'])
plt.ylabel('Total Host Walltime (s)')
plt.title('Total Host Walltime')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('img/total_host_walltime.png')
plt.close()

# Quantized vs Full-Precision Improvements
for device in df['device'].unique():
    full = df[(df['device'] == device) & (df['precision'] == 'Full-Precision')]
    quant = df[(df['device'] == device) & (df['precision'] == 'Quantized')]
    if not full.empty and not quant.empty:
        thr_imp = (quant['throughput_qps'].iloc[0] / full['throughput_qps'].iloc[0] - 1) * 100
        lat_imp = (full['latency_mean_ms'].iloc[0] - quant['latency_mean_ms'].iloc[0]) / full['latency_mean_ms'].iloc[0] * 100
        print(f"{device}: Throughput improvement = {thr_imp:.1f}%, Mean latency reduction = {lat_imp:.1f}%")

