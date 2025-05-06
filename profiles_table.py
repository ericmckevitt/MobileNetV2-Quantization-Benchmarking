import json
import glob
import pandas as pd

# Load all JSONs into a flat DataFrame
records = []
for path in glob.glob('data/*/*.json'):
    device = path.split('/')[-2]
    precision = 'int8' if 'int8' in path else 'fp32'

    with open(path) as f:
        for entry in json.load(f):
            if 'name' in entry and 'timeMs' in entry:
                records.append({
                    'device': device,
                    'precision': precision,
                    'layer': entry['name'],
                    'timeMs': entry['timeMs']
                })

df = pd.DataFrame(records)
pivot = df.pivot_table(
    index=['device','layer'],
    columns='precision',
    values='timeMs'
).reset_index()

# Compute speedup ratio = FP32 time / INT8 time
pivot['speedup_x'] = pivot['fp32'] / pivot['int8']

# Print top-10 speedups for each device
for device in ['orin_nx', 'xavier_agx']:
    dev_df = pivot[pivot.device == device] \
                .sort_values('speedup_x', ascending=False) \
                .head(10)[['layer','fp32','int8','speedup_x']]
    print(f"\n=== Top 10 INT8 speedups on {device.replace('_',' ').title()} ===")
    print(dev_df.to_string(index=False, 
                           formatters={
                               'fp32': '{:8.2f}ms'.format,
                               'int8': '{:8.2f}ms'.format,
                               'speedup_x': '{:6.2f}x'.format
                           }))
