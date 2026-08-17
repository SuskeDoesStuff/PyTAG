# Exporter for PyTAG PPO (PPONet) policies.

# Reads a trained agent.pt and writes a plain-JSON file describing the MLP: obs -> [Linear, ReLU] x N  ->  actor Linear (logits) plus the critic head (exported for later use as a value heuristic).

import re
import sys
import json
import numpy as np
import torch


def main():
    if len(sys.argv) != 3:
        print("usage: python export_weights.py <agent.pt> <out.txt>")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    sd = torch.load(in_path, map_location="cpu")
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    
    for k, v in sd.items():
        if v.ndim == 4:
            sys.exit(f"ERROR: '{k}' is a conv weight (4D). This exporter only "
                     f"handles MLP policies. Use ONNX for CNN/LSTM policies.")

    def npd(key):
        return sd[key].detach().cpu().numpy().astype(np.float64)

    # Collect the trunk's Linear layers in order (network.0, network.2, ...).
    hidden_idx = sorted({
        int(m.group(1)) for k in sd
        if (m := re.match(r"network\.(\d+)\.weight$", k)) and sd[k].ndim == 2
    })
    hidden = [{"w": npd(f"network.{i}.weight").tolist(),
               "b": npd(f"network.{i}.bias").tolist()} for i in hidden_idx]

    actor = {"w": npd("actor.weight").tolist(), "b": npd("actor.bias").tolist()}
    critic = {"w": npd("critic.weight").tolist(), "b": npd("critic.bias").tolist()}

    obs_dim = len(hidden[0]["w"][0])
    n_actions = len(actor["w"])
    hidden_sizes = [len(h["w"]) for h in hidden]
    meta = {"obs_dim": obs_dim, "n_actions": n_actions, "hidden_sizes": hidden_sizes}
    print("meta:", meta)

    with open(out_path, "w") as f:
        json.dump({"hidden": hidden, "actor": actor, "critic": critic, "meta": meta}, f)
    print(f"wrote {out_path}")

    # Reference forward pass
    Ws = [np.array(h["w"]) for h in hidden]
    bs = [np.array(h["b"]) for h in hidden]
    aW, aB = np.array(actor["w"]), np.array(actor["b"])

    def forward(obs):
        x = obs
        for W, b in zip(Ws, bs):
            x = np.maximum(W @ x + b, 0.0)
        return aW @ x + aB

    tests = {
        "zeros": np.zeros(obs_dim),
        "ones": np.ones(obs_dim),
        "ramp": np.linspace(-1, 1, obs_dim),
    }
    print("\n--- reference outputs ---")
    for name, obs in tests.items():
        logits = forward(obs)
        print(f"{name}: argmax={int(np.argmax(logits))}  "
              f"logits={[round(float(x), 4) for x in logits]}")


if __name__ == "__main__":
    main()
