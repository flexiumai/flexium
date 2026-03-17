#!/usr/bin/env python
"""Demo of GPU error recovery with flexium.

Usage:
    python examples/simple/oom_recovery_demo.py --mode simple
    python examples/simple/oom_recovery_demo.py --mode decorator
    python examples/simple/oom_recovery_demo.py --mode iterator
"""

import argparse
import multiprocessing
import time
import torch
import flexium.auto


def gpu_filler(device: int, fill_pct: int, ready_event, stop_event):
    """Fill GPU memory in a subprocess."""
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / 1e9
    fill_gb = total_gb * fill_pct / 100

    filler = torch.zeros(int(fill_gb * 1e9 / 4), dtype=torch.float32, device=f"cuda:{device}")
    ready_event.set()  # Signal that GPU is filled
    stop_event.wait()  # Wait until main process says stop
    del filler


def demo_simple(total_gb: float) -> None:
    """Simple context manager - operation is LOST on error."""
    print("\n=== SIMPLE MODE: Operation is LOST, training continues ===")
    print("    (The failed batch is skipped, training continues with next batch)\n")

    batches = [
        {"id": 1, "data": torch.randn(1000, 1000)},
        {"id": 2, "data": torch.randn(1000, 1000)},
        {"id": 3, "data": torch.randn(1000, 1000)},  # This one will OOM
        {"id": 4, "data": torch.randn(1000, 1000)},
        {"id": 5, "data": torch.randn(1000, 1000)},
    ]

    print(f"Processing {len(batches)} batches...")
    print("Batch 3 will trigger OOM by allocating 30% of GPU\n")

    input("Press Enter to start processing batches...")

    for batch in batches:
        print(f"\n[Batch {batch['id']}] Processing on {flexium.auto.get_physical_device()}...")

        with flexium.auto.recoverable():
            if batch["id"] == 3:
                print(f"[Batch {batch['id']}] Allocating 30% of GPU (will OOM)...")
                big_tensor = torch.zeros(int(total_gb * 0.3 * 1e9 / 4), dtype=torch.float32, device="cuda")
                result = big_tensor.sum().item()
                del big_tensor
            else:
                data = batch["data"].cuda()
                result = data.sum().item()
                del data

            print(f"[Batch {batch['id']}] Result: {result:.2f}")

        print(f"[Batch {batch['id']}] Done (device: {flexium.auto.get_physical_device()})")

    print("\n=== RESULT: Batch 3 was LOST but batches 4,5 continued on new GPU ===")


def demo_decorator(total_gb: float) -> None:
    """Decorator - operation is RETRIED on new GPU with SAME data."""
    print("\n=== DECORATOR MODE: Operation is RETRIED with same data ===")
    print("    (The exact same function call is replayed on new GPU)\n")

    batch_data = torch.randn(1000, 1000)
    batch_sum = batch_data.sum().item()
    print(f"Batch data created. CPU sum: {batch_sum:.4f}")
    print("(We'll verify this same sum after migration)\n")

    @flexium.auto.recoverable(retries=3)
    def process_batch(data: torch.Tensor, trigger_oom: bool) -> float:
        device = flexium.auto.get_physical_device()
        print(f"   process_batch() on {device}, trigger_oom={trigger_oom}")

        if trigger_oom:
            print("   Allocating 30% of GPU to trigger OOM...")
            _ = torch.zeros(int(total_gb * 0.3 * 1e9 / 4), dtype=torch.float32, device="cuda")

        gpu_data = data.cuda()
        result = gpu_data.sum().item()
        print(f"   Computed sum: {result:.4f}")
        return result

    print(f"Current device: {flexium.auto.get_physical_device()}")
    input("\nPress Enter to process batch (will OOM, migrate, and REPLAY)...")

    result = process_batch(batch_data, trigger_oom=True)

    print("\n=== RESULT ===")
    print(f"   Expected sum: {batch_sum:.2f}")
    print(f"   Got sum:      {result:.2f}")
    print(f"   Match: {round(batch_sum, 2) == round(result, 2)}")
    print("   The SAME data was replayed on the new GPU!")


def demo_iterator(total_gb: float) -> None:
    """Iterator - you control the retry loop with SAME data."""
    print("\n=== ITERATOR MODE: You control retry with same data ===")
    print("    (You explicitly loop and can verify data between attempts)\n")

    batch_data = torch.randn(1000, 1000)
    batch_sum = batch_data.sum().item()
    print(f"Batch data created. CPU sum: {batch_sum:.4f}")
    print("(We'll verify this same sum after migration)\n")

    print(f"Current device: {flexium.auto.get_physical_device()}")
    input("\nPress Enter to process batch (will OOM, migrate, and RETRY)...")

    attempt = 0
    for recovery_attempt in flexium.auto.recoverable(retries=3):
        with recovery_attempt:
            attempt += 1
            device = flexium.auto.get_physical_device()
            print(f"\n   Attempt {attempt} on {device}")

            if attempt == 1:
                print("   Allocating 30% of GPU to trigger OOM...")
                _ = torch.zeros(int(total_gb * 0.3 * 1e9 / 4), dtype=torch.float32, device="cuda")

            gpu_data = batch_data.cuda()
            result = gpu_data.sum().item()
            print(f"   Computed sum: {result:.4f}")

    print("\n=== RESULT ===")
    print(f"   Expected sum: {batch_sum:.2f}")
    print(f"   Got sum:      {result:.2f}")
    print(f"   Match: {round(batch_sum, 2) == round(result, 2)}")
    print("   The SAME data was processed on the new GPU!")


def main() -> None:
    parser = argparse.ArgumentParser(description="OOM recovery demo")
    parser.add_argument("--mode", choices=["simple", "decorator", "iterator"], default="simple")
    parser.add_argument("--fill-pct", type=int, default=80, help="GPU fill percentage (default: 80)")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9

    print(f"GPUs: {torch.cuda.device_count()}, GPU 0 memory: {total_gb:.1f} GB")
    print("-" * 60)

    # Start GPU filler subprocess
    ready_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()
    filler_proc = multiprocessing.Process(target=gpu_filler, args=(0, args.fill_pct, ready_event, stop_event))
    filler_proc.start()

    print(f"Filling cuda:0 with {args.fill_pct}% memory...")
    ready_event.wait()
    print(f"GPU filled. Free: ~{total_gb * (100 - args.fill_pct) / 100:.1f} GB")
    print("-" * 60)

    try:
        with flexium.auto.run(orchestrator=""):
            if args.mode == "simple":
                demo_simple(total_gb)
            elif args.mode == "decorator":
                demo_decorator(total_gb)
            else:
                demo_iterator(total_gb)

            device = flexium.auto.get_physical_device()
            print(f"\nFinal device: {device}")
            input(f"\nPress Enter to exit (check nvidia-smi shows process on {device})...")
    finally:
        stop_event.set()
        filler_proc.join(timeout=2)
        if filler_proc.is_alive():
            filler_proc.terminate()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
