#!/usr/bin/env python
"""Helper script to fill GPU memory.

Usage:
    python examples/simple/gpu_filler.py --device 0 --fill-pct 80
"""

import argparse
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill GPU memory")
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--fill-pct", type=int, default=80, help="Percentage to fill")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    total_gb = props.total_memory / 1e9
    fill_gb = total_gb * args.fill_pct / 100

    print(f"Filling cuda:{args.device} with {fill_gb:.1f} GB ({args.fill_pct}%)...")
    filler = torch.zeros(int(fill_gb * 1e9 / 4), dtype=torch.float32, device=f"cuda:{args.device}")
    print(f"Done. Press Ctrl+C to release memory and exit.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        del filler
        print("\nReleased.")


if __name__ == "__main__":
    main()
