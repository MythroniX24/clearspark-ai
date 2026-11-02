"""
Exports WAL-Net trained weights into JSON format for Web Worker / Client-Side execution.
"""

import numpy as np
import json
import os

def export_weights(weights_dir, out_json_path):
    wal_path = os.path.join(weights_dir, "wal_net_weights.npy")
    if not os.path.exists(wal_path):
        print(f"Weights file not found at {wal_path}")
        return False

    state = np.load(wal_path, allow_pickle=True).item()
    web_state = {}
    for k, v in state.items():
        # Store as rounded float32 list
        web_state[k] = {
            'shape': list(v.shape),
            'data': np.round(v.flatten(), 4).tolist()
        }

    with open(out_json_path, "w") as f:
        json.dump(web_state, f)

    size_kb = os.path.getsize(out_json_path) / 1024.0
    print(f"Exported web weights to {out_json_path} ({size_kb:.1f} KB)")
    return True

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    w_dir = os.path.join(base_dir, "weights")
    out_file = os.path.join(base_dir, "weights", "sparklenet_web_weights.json")
    export_weights(w_dir, out_file)
