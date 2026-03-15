# Examples

This page provides comprehensive examples of using Flexium.AI in various scenarios.

## Table of Contents

1. [Before & After: Real-World Scenarios](#before-after-real-world-scenarios)
2. [Basic Examples](#basic-examples)
3. [MNIST Training](#mnist-training)
4. [PyTorch Lightning](#pytorch-lightning)
5. [ResNet Training](#resnet-training)
6. [Multi-GPU Workflows](#multi-gpu-workflows)
7. [Production Patterns](#production-patterns)
8. [Advanced Examples](#advanced-examples)
    - [GAN Training (DCGAN)](#gan-training-dcgan)
    - [Diffusion Model Training (DDPM)](#diffusion-model-training-ddpm)
    - [Transformer Training (GPT-style)](#transformer-training-gpt-style)
    - [Vision Transformer (ViT)](#vision-transformer-vit-training)
9. [Zero-Residue Migration](#zero-residue-migration)

---

## Before & After: Real-World Scenarios

These examples show the pain points of GPU management **without** Flexium.AI and how Flexium.AI solves them.

### Scenario 1: GPU Contention (Need to Free a GPU)

**The Problem:** You're training a model on `cuda:0`, but a colleague needs that GPU for an urgent deadline. Without Flexium.AI, you have to stop your training, lose progress, and restart later.

=== "❌ Without flexium"

    ```python
    # train.py - Running on cuda:0
    import torch

    model = Net().cuda()  # On cuda:0
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(100):
        for batch in dataloader:
            # ... training ...
            pass
        print(f"Epoch {epoch} complete")

    # Colleague needs cuda:0 NOW!
    # Options:
    #   1. Kill the process (lose progress since last manual checkpoint)
    #   2. Wait (colleague misses deadline)
    #   3. Try model.to("cuda:1") (leaves memory on cuda:0!)
    ```

    **What happens when you try to move:**
    ```python
    # Attempt to free cuda:0
    model = model.to("cuda:1")
    torch.cuda.empty_cache()

    # nvidia-smi STILL shows memory on cuda:0!
    # PyTorch's caching allocator holds onto memory
    # CUDA context overhead remains
    # Your colleague still can't use the GPU
    ```

=== "✅ With flexium"

    ```python
    # train.py - Same code, just add 2 lines
    import flexium.auto
    import torch

    with flexium.auto.run():
        model = Net().cuda()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(100):
            for batch in dataloader:
                # ... training ...
                pass
            print(f"Epoch {epoch} complete")
    ```

    **When colleague needs the GPU:**
    ```bash
    # One command - training continues on cuda:1
    flexium-ctl migrate gpu-abc123 cuda:1

    # nvidia-smi shows cuda:0 is 100% FREE
    # Training resumed from exact batch on cuda:1
    # No progress lost!
    ```

---

### Scenario 2: Out of Memory (OOM) Error

**The Problem:** Your training crashes at 3 AM with OOM. You lose hours of training progress and have to restart manually.

=== "❌ Without flexium"

    ```python
    # Long-running training job
    model = LargeModel().cuda()

    for epoch in range(100):
        for batch in dataloader:
            # At epoch 47, batch 892... CRASH!
            # RuntimeError: CUDA out of memory.
            # Tried to allocate 2.00 GiB
            pass

    # Result:
    # - Training crashed at 3 AM
    # - Lost 47 epochs of progress (unless you had manual checkpoints)
    # - You wake up to a failed job
    # - Have to manually restart, find a GPU with more VRAM
    ```

=== "✅ With flexium"

    ```python
    import flexium.auto

    with flexium.auto.run():
        model = LargeModel().cuda()

        for epoch in range(100):
            for batch in dataloader:
                # At epoch 47, batch 892... OOM detected!
                # flexium can:
                # 1. Detect the OOM error
                # 2. Migrate to GPU with more VRAM
                # 3. Restore training state
                # 4. Continue training
                pass

    # Result:
    # - Training continues on new device
    # - You wake up to a completed job
    # - Minimal manual intervention needed
    ```

    **What you see in logs:**
    ```
    [flexium] GPU Out of Memory detected
    [flexium]   Memory in use: 14.2 GB
    [flexium]   Tried to allocate: 2.0 GB
    [flexium] Requesting migration...
    [flexium] Migrating to cuda:2 (24GB VRAM)
    [flexium] Training resumed
    ```

---

### Scenario 3: Shared GPU Cluster

**The Problem:** Your team shares 8 GPUs. Jobs compete for resources, there's no visibility into who's using what, and priority jobs can't preempt less important ones.

=== "❌ Without flexium"

    ```bash
    # Alice starts training on cuda:0
    $ python alice_train.py  # Uses cuda:0

    # Bob starts training, doesn't know cuda:0 is used
    $ python bob_train.py  # Also tries cuda:0, OOM!

    # Charlie has urgent deadline, needs cuda:0
    # Has to Slack Alice: "Hey can you stop your job?"
    # Alice is in a meeting, doesn't respond for 2 hours
    # Charlie misses deadline

    # No visibility:
    $ nvidia-smi  # Shows PIDs but not who owns them or their progress
    ```

=== "✅ With flexium"

    ```bash
    # Start orchestrator with dashboard (once)
    $ flexium-ctl server --dashboard

    # Alice starts training
    $ python alice_train.py  # Auto-registers with orchestrator

    # Bob starts training - orchestrator assigns cuda:1
    $ python bob_train.py  # Gets cuda:1 automatically

    # Charlie has urgent deadline
    $ flexium-ctl list
    PROCESS         DEVICE    STATUS    HOSTNAME
    alice-abc123    cuda:0    running   node01
    bob-def456      cuda:1    running   node01

    # Charlie migrates Alice's job (no interruption)
    $ flexium-ctl migrate alice-abc123 cuda:2

    # Alice's training continues on cuda:2
    # cuda:0 is now free for Charlie
    # No Slack messages needed!
    ```

    **Dashboard view (http://localhost:8080):**
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │  Flexium.AI Dashboard                                    │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  cuda:0 (Tesla V100 32GB)          cuda:1 (Tesla V100 32GB) │
    │  ┌──────────────────────┐          ┌──────────────────────┐ │
    │  │ alice-abc123         │          │ bob-def456           │ │
    │  │ Status: running      │          │ Status: running      │ │
    │  │ VRAM: 8.2/32 GB      │          │ VRAM: 12.1/32 GB     │ │
    │  │ [Migrate] [Details]  │          │ [Migrate] [Details]  │ │
    │  └──────────────────────┘          └──────────────────────┘ │
    │                                                              │
    │  cuda:2 (Tesla V100 32GB)          cuda:3 (Tesla V100 32GB) │
    │  ┌──────────────────────┐          ┌──────────────────────┐ │
    │  │ (available)          │          │ (available)          │ │
    │  └──────────────────────┘          └──────────────────────┘ │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    ```

---

### Scenario 4: Hardware Failure

**The Problem:** A GPU develops ECC errors or fails during training. Your job crashes, you lose progress, and you have to manually identify the problem and restart.

=== "❌ Without flexium"

    ```python
    # Training on cuda:2
    model = Net().cuda()

    for epoch in range(100):
        for batch in dataloader:
            # At epoch 23...
            # RuntimeError: CUDA error: uncorrectable ECC error
            #
            # What now?
            # 1. Check nvidia-smi - which GPU failed?
            # 2. Manually restart on different GPU
            # 3. Hope you had a recent checkpoint
            # 4. Debug for hours figuring out the problem
            pass
    ```

=== "✅ With flexium"

    ```python
    import flexium.auto

    with flexium.auto.run():
        model = Net().cuda()

        for epoch in range(100):
            for batch in dataloader:
                # At epoch 23... ECC error detected!
                # flexium automatically:
                # 1. Catches the ECC error
                # 2. Marks cuda:2 as unhealthy
                # 3. Migrates to healthy GPU
                # 4. Continues training
                pass

    # Dashboard shows cuda:2 with red "Unhealthy" badge
    # Other jobs won't be scheduled there
    # Admin can investigate and mark healthy when fixed
    ```

---

### Scenario 5: Preemption for Priority Jobs

**The Problem:** An urgent inference job or deadline-critical training needs a GPU immediately, but all GPUs are occupied. You have to interrupt colleagues, wait, or miss your deadline.

=== "❌ Without flexium"

    ```bash
    # All 4 GPUs are in use
    $ nvidia-smi
    # cuda:0 - PID 12345 (whose job? what priority? no idea)
    # cuda:1 - PID 12346
    # cuda:2 - PID 12347
    # cuda:3 - PID 12348

    # You need a GPU NOW for urgent inference demo
    # Options:
    #   1. Slack everyone: "Who can stop their job?"
    #   2. Wait (miss the demo)
    #   3. Kill a random process (someone loses work)
    #   4. Run on CPU (too slow for demo)

    # 30 minutes later, someone responds...
    # Demo already failed.
    ```

=== "✅ With flexium"

    ```bash
    # See all jobs with context
    $ flexium-ctl list
    PROCESS           DEVICE    STATUS    HOSTNAME
    alice-research    cuda:0    running   node01
    bob-experiment    cuda:1    running   node01
    charlie-train     cuda:2    running   node01
    dave-baseline     cuda:3    running   node01

    # Check dashboard for memory usage - dave just started (low VRAM)
    # dave-baseline just started - easy to pause and restart later

    # Pause dave's job (frees GPU completely)
    $ flexium-ctl pause dave-baseline

    # cuda:3 is now FREE - run your urgent demo
    $ python urgent_inference.py --device cuda:3

    # After demo, resume dave back to GPU
    $ flexium-ctl resume dave-baseline cuda:3

    # Result:
    # - Demo succeeded
    # - Dave's job paused briefly, then continued from exact same point
    # - No Slack messages, no waiting, no lost work
    ```

    **Dashboard shows memory usage at a glance:**
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │  Flexium.AI Dashboard                       [Preempt]   │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  cuda:0 - alice-research        cuda:1 - bob-experiment     │
    │  ├─ Status: running             ├─ Status: running          │
    │  └─ VRAM: 24/32 GB              └─ VRAM: 16/32 GB           │
    │                                                              │
    │  cuda:2 - charlie-train         cuda:3 - dave-baseline      │
    │  ├─ Status: running             ├─ Status: running ← NEW    │
    │  └─ VRAM: 28/32 GB              └─ VRAM: 8/32 GB            │
    │                                                              │
    │  → dave-baseline has low VRAM usage, safe to pause          │
    └─────────────────────────────────────────────────────────────┘
    ```

---

### Scenario 6: Long-Running Experiments

**The Problem:** You're running a 2-week training job. Various things can go wrong: server reboots, driver updates, competing jobs, etc.

=== "❌ Without flexium"

    ```python
    # 2-week training job
    model = BigModel().cuda()

    for epoch in range(1000):
        for batch in dataloader:
            # Day 3: Server reboots for kernel update
            #        -> Job killed, restart from scratch (or last manual checkpoint)
            #
            # Day 7: Colleague needs your GPU urgently
            #        -> Have to stop, lose progress
            #
            # Day 10: OOM due to memory fragmentation
            #         -> Job crashes, restart manually
            #
            # Day 12: GPU develops ECC errors
            #         -> Job crashes, debug for hours
            pass

    # Result: 2-week job actually takes 4 weeks due to failures
    ```

=== "✅ With flexium"

    ```python
    import flexium.auto

    with flexium.auto.run(orchestrator="orchestrator:80"):
        model = BigModel().cuda()

        for epoch in range(1000):
            for batch in dataloader:
                # Day 3: Server reboots
                #        -> Checkpoint saved, job resumes after reboot
                #
                # Day 7: Colleague needs GPU
                #        -> Admin migrates job to cuda:2, continues without interruption
                #
                # Day 10: OOM detected
                #         -> Migrate to GPU with more VRAM
                #
                # Day 12: ECC error
                #         -> Migrate to healthy GPU
                pass

    # Result: 2-week job completes in ~2 weeks
    # Failures handled with migration
    ```

---

### Summary: What Flexium.AI Gives You

| Scenario | Without Flexium.AI | With Flexium.AI |
|----------|---------------------|------------------|
| GPU contention | Stop job, lose progress | Live migration, zero downtime |
| OOM error | Job crashes, manual restart | Auto-recovery to bigger GPU |
| Shared cluster | Slack messages, conflicts | Dashboard, CLI, organized |
| Hardware failure | Debug + manual restart | Auto-migrate, mark unhealthy |
| Priority preemption | Interrupt people, wait, miss deadline | Instant migration, no lost work |
| Long jobs | Multiple failures | Resilient, auto-recovery |

---

## Basic Examples

### Minimal Example

The simplest way to add flexium:

??? example "Minimal Training Example"

    ```python
    import flexium.auto
    import torch
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 10)

        def forward(self, x):
            return self.fc(x)

    with flexium.auto.run():
        # Everything inside is standard PyTorch
        model = SimpleNet().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for i in range(1000):
            x = torch.randn(32, 100).cuda()
            y = torch.randint(0, 10, (32,)).cuda()

            output = model(x)
            loss = nn.functional.cross_entropy(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss.item():.4f}")
    ```

---

## MNIST Training

### Basic MNIST Training (mnist_train_auto.py)

??? example "Complete MNIST Training Script"

    ```python
    #!/usr/bin/env python
    """MNIST training with transparent flexium."""

    import argparse
    import time

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    import flexium.auto


    class Net(nn.Module):
        """Simple CNN for MNIST."""

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)


    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--orchestrator", default=None)
        parser.add_argument("--device", default=None)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--disabled", action="store_true")
        args = parser.parse_args()

        with flexium.auto.run(
            orchestrator=args.orchestrator,
            device=args.device,
            disabled=args.disabled,
        ):
            # Data
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_data = datasets.MNIST(
                "./data", train=True, download=True, transform=transform
            )
            train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

            # Model - just use .cuda()!
            model = Net().cuda()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop - completely standard
            for epoch in range(args.epochs):
                epoch_start = time.time()
                total_loss = 0
                correct = 0
                total = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.cuda(), target.cuda()

                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)

                    if batch_idx % 200 == 0:
                        print(f"Epoch {epoch:2d} | Batch {batch_idx:4d} | "
                              f"Loss: {loss.item():.4f} | "
                              f"Acc: {100.*correct/total:.1f}%")

                epoch_time = time.time() - epoch_start
                print(f">>> Epoch {epoch} done | "
                      f"Avg Loss: {total_loss/len(train_loader):.4f} | "
                      f"Acc: {100.*correct/total:.1f}% | "
                      f"Time: {epoch_time:.2f}s\n")


    if __name__ == "__main__":
        main()
    ```

---

## PyTorch Lightning

Flexium integrates seamlessly with PyTorch Lightning using the `FlexiumCallback`.

### Quick Start

```python
from pytorch_lightning import Trainer
from flexium.lightning import FlexiumCallback

# Just add the callback - that's it!
trainer = Trainer(
    callbacks=[FlexiumCallback(orchestrator="localhost:80")],
    max_epochs=100,
    accelerator="gpu",
    devices=1,
)
trainer.fit(model, dataloader)
```

### Complete MNIST Example with Lightning

??? example "Complete MNIST Lightning Script"

    ```python
    #!/usr/bin/env python
    """MNIST training with PyTorch Lightning and Flexium."""

    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    from flexium.lightning import FlexiumCallback


    class MNISTModel(pl.LightningModule):
        """Simple CNN for MNIST classification."""

        def __init__(self, learning_rate: float = 0.001):
            super().__init__()
            self.save_hyperparameters()

            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

        def training_step(self, batch, batch_idx):
            data, target = batch
            output = self(data)
            loss = F.nll_loss(output, target)

            # Calculate accuracy
            pred = output.argmax(dim=1)
            acc = (pred == target).float().mean()

            self.log("train_loss", loss, prog_bar=True)
            self.log("train_acc", acc, prog_bar=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


    class MNISTDataModule(pl.LightningDataModule):
        """DataModule for MNIST dataset."""

        def __init__(self, data_dir="./data", batch_size=64):
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

        def prepare_data(self):
            datasets.MNIST(self.data_dir, train=True, download=True)

        def setup(self, stage=None):
            self.train_dataset = datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )

        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


    def main():
        # Set seed for reproducibility
        pl.seed_everything(42)

        # Create model and data
        model = MNISTModel()
        datamodule = MNISTDataModule()

        # === THIS IS THE ONLY CHANGE FOR FLEXIUM ===
        flexium_callback = FlexiumCallback(
            orchestrator="localhost:80",  # Or use FLEXIUM_SERVER env var
        )

        # Create trainer with Flexium callback
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator="gpu",
            devices=1,
            callbacks=[flexium_callback],
        )

        # Train - migration happens transparently!
        trainer.fit(model, datamodule)


    if __name__ == "__main__":
        main()
    ```

### Running the Lightning Example

```bash
# Start orchestrator
flexium-ctl server --dashboard

# Run Lightning example
python examples/lightning/mnist_lightning.py

# With custom settings
python examples/lightning/mnist_lightning.py --orchestrator localhost:80 --epochs 5

# Baseline (no flexium)
python examples/lightning/mnist_lightning.py --disabled
```

### Comparison: Raw PyTorch vs Lightning

=== "Raw PyTorch"

    ```python
    import flexium.auto

    with flexium.auto.run(orchestrator="localhost:80"):
        model = Net().cuda()
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(10):
            for batch in dataloader:
                data, target = batch[0].cuda(), batch[1].cuda()
                # ... training loop ...
    ```

=== "PyTorch Lightning"

    ```python
    from flexium.lightning import FlexiumCallback

    trainer = Trainer(
        callbacks=[FlexiumCallback(orchestrator="localhost:80")],
        max_epochs=10,
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, dataloader)
    ```

Both approaches provide the same transparent migration capability. Choose based on your preference:

- **Raw PyTorch**: More control, minimal dependencies
- **Lightning**: Less boilerplate, built-in features (logging, checkpointing, etc.)

### FlexiumCallback Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `orchestrator` | `str` | `None` | Orchestrator address (host:port) |
| `device` | `str` | `None` | Initial device (auto-detected if not set) |
| `disabled` | `bool` | `False` | Disable Flexium for debugging |

### Installation

```bash
# Install Flexium with Lightning support
pip install flexium[lightning]

# Or install Lightning separately
pip install pytorch-lightning>=2.0.0
```

For more details, see [Lightning Integration](features/lightning-integration.md).

---

## ResNet Training

### ImageNet-style Training with flexium

??? example "ResNet-50 ImageNet Training Script"

    ```python
    #!/usr/bin/env python
    """ResNet training with flexium."""

    import flexium.auto
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms


    def main():
        with flexium.auto.run():
            # Model
            model = models.resnet50(pretrained=False).cuda()
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

            # Data
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            dataset = datasets.ImageFolder("/path/to/imagenet/train", transform)
            dataloader = DataLoader(
                dataset,
                batch_size=256,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
            )

            # Training
            model.train()
            for epoch in range(90):
                for i, (images, target) in enumerate(dataloader):
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                    output = model(images)
                    loss = criterion(output, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % 100 == 0:
                        print(f"Epoch [{epoch}][{i}/{len(dataloader)}] "
                              f"Loss: {loss.item():.4f}")

                scheduler.step()


    if __name__ == "__main__":
        main()
    ```

---

## Multi-GPU Workflows

### Coordinated Training Jobs

Run multiple training jobs and migrate between them:

```python
# job1.py
import flexium.auto

with flexium.auto.run(orchestrator="orchestrator:80"):
    # Training job 1
    model1 = Model1().cuda()
    train(model1)

# job2.py
import flexium.auto

with flexium.auto.run(orchestrator="orchestrator:80"):
    # Training job 2
    model2 = Model2().cuda()
    train(model2)
```

Then use CLI to manage:

```bash
# See both jobs
flexium-ctl list

# Move job1 to cuda:0, job2 to cuda:1
flexium-ctl migrate job1-process-id cuda:0
flexium-ctl migrate job2-process-id cuda:1
```

---

## Production Patterns

### With Error Handling

??? example "Error Handling Pattern"

    ```python
    import flexium.auto
    import torch
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    def train():
        with flexium.auto.run(orchestrator="prod-orchestrator:80"):
            model = MyModel().cuda()
            optimizer = torch.optim.Adam(model.parameters())

            for epoch in range(100):
                try:
                    for batch in dataloader:
                        # Training step
                        loss = train_step(model, optimizer, batch)

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.error("OOM error - consider migrating to GPU with more memory")
                        raise
                    else:
                        raise


    if __name__ == "__main__":
        train()
    ```

### With Checkpointing

??? example "Checkpointing Pattern"

    ```python
    import flexium.auto
    import torch
    from pathlib import Path


    def train():
        checkpoint_dir = Path("./checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        with flexium.auto.run():
            model = MyModel().cuda()
            optimizer = torch.optim.Adam(model.parameters())
            start_epoch = 0

            # Resume from checkpoint if exists
            checkpoint_path = checkpoint_dir / "latest.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {start_epoch}")

            for epoch in range(start_epoch, 100):
                train_epoch(model, optimizer, dataloader)

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, checkpoint_path)


    if __name__ == "__main__":
        train()
    ```

### With Distributed Training (Future)

??? example "Distributed Training (Future)"

    ```python
    # Note: Multi-machine distributed training is a future enhancement
    # For now, use single-machine, single-GPU per process

    import flexium.auto

    # Each process runs independently
    with flexium.auto.run():
        model = MyModel().cuda()
        train(model)
    ```

---

## Advanced Examples

These examples demonstrate flexium with more complex models used in real-world ML research.

### GAN Training (DCGAN)

Training a Deep Convolutional GAN on CIFAR-10:

??? example "DCGAN Training Script"

    ```python
    #!/usr/bin/env python
    """DCGAN training with flexium.

    A Deep Convolutional GAN trained on CIFAR-10.
    Demonstrates handling of two models (generator + discriminator)
    and alternating optimization steps.
    """

    import flexium.auto
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import save_image


    # Generator network
    class Generator(nn.Module):
        def __init__(self, latent_dim=100, channels=3, features=64):
            super().__init__()
            self.main = nn.Sequential(
                # Input: latent_dim x 1 x 1
                nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(features * 8),
                nn.ReLU(True),
                # State: (features*8) x 4 x 4
                nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.ReLU(True),
                # State: (features*4) x 8 x 8
                nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.ReLU(True),
                # State: (features*2) x 16 x 16
                nn.ConvTranspose2d(features * 2, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
                # Output: channels x 32 x 32
            )

        def forward(self, z):
            return self.main(z.view(z.size(0), -1, 1, 1))


    # Discriminator network
    class Discriminator(nn.Module):
        def __init__(self, channels=3, features=64):
            super().__init__()
            self.main = nn.Sequential(
                # Input: channels x 32 x 32
                nn.Conv2d(channels, features, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # State: features x 16 x 16
                nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # State: (features*2) x 8 x 8
                nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(features * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # State: (features*4) x 4 x 4
                nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.main(x).view(-1)


    def main():
        latent_dim = 100
        batch_size = 128
        epochs = 100
        lr = 0.0002

        with flexium.auto.run():
            # Data
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            dataset = datasets.CIFAR10(
                "./data", train=True, download=True, transform=transform
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )

            # Models - both go to cuda
            generator = Generator(latent_dim).cuda()
            discriminator = Discriminator().cuda()

            # Optimizers
            opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
            opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

            # Loss
            criterion = nn.BCELoss()

            # Fixed noise for visualization
            fixed_noise = torch.randn(64, latent_dim).cuda()

            for epoch in range(epochs):
                for i, (real_images, _) in enumerate(dataloader):
                    batch_size = real_images.size(0)
                    real_images = real_images.cuda()

                    # Labels
                    real_labels = torch.ones(batch_size).cuda()
                    fake_labels = torch.zeros(batch_size).cuda()

                    # ---------------------
                    # Train Discriminator
                    # ---------------------
                    opt_d.zero_grad()

                    # Real images
                    output_real = discriminator(real_images)
                    loss_d_real = criterion(output_real, real_labels)

                    # Fake images
                    noise = torch.randn(batch_size, latent_dim).cuda()
                    fake_images = generator(noise)
                    output_fake = discriminator(fake_images.detach())
                    loss_d_fake = criterion(output_fake, fake_labels)

                    loss_d = loss_d_real + loss_d_fake
                    loss_d.backward()
                    opt_d.step()

                    # ---------------------
                    # Train Generator
                    # ---------------------
                    opt_g.zero_grad()

                    output = discriminator(fake_images)
                    loss_g = criterion(output, real_labels)  # Want D to think fake is real

                    loss_g.backward()
                    opt_g.step()

                    if i % 100 == 0:
                        print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                              f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f}")

                # Save sample images
                with torch.no_grad():
                    fake = generator(fixed_noise)
                    save_image(fake, f"samples/epoch_{epoch:03d}.png", normalize=True)


    if __name__ == "__main__":
        main()
    ```

---

### Diffusion Model Training (DDPM)

Training a Denoising Diffusion Probabilistic Model:

??? example "DDPM Training Script"

    ```python
    #!/usr/bin/env python
    """DDPM (Denoising Diffusion) training with flexium.

    A simplified implementation of DDPM for image generation.
    Demonstrates handling of complex training loops with
    timestep conditioning and noise scheduling.
    """

    import math
    import flexium.auto
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms


    class SinusoidalPosEmb(nn.Module):
        """Sinusoidal positional embeddings for timestep conditioning."""

        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, t):
            device = t.device
            half_dim = self.dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
            emb = t[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            return emb


    class ResBlock(nn.Module):
        """Residual block with time conditioning."""

        def __init__(self, in_ch, out_ch, time_emb_dim):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)
            self.norm1 = nn.GroupNorm(8, in_ch)
            self.norm2 = nn.GroupNorm(8, out_ch)

            if in_ch != out_ch:
                self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
            else:
                self.shortcut = nn.Identity()

        def forward(self, x, t_emb):
            h = F.silu(self.norm1(x))
            h = self.conv1(h)
            h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
            h = F.silu(self.norm2(h))
            h = self.conv2(h)
            return h + self.shortcut(x)


    class UNet(nn.Module):
        """Simple UNet for diffusion model."""

        def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
            super().__init__()

            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(base_channels),
                nn.Linear(base_channels, time_emb_dim),
                nn.GELU(),
                nn.Linear(time_emb_dim, time_emb_dim),
            )

            # Encoder
            self.enc1 = ResBlock(in_channels, base_channels, time_emb_dim)
            self.enc2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
            self.enc3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)

            # Middle
            self.mid = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)

            # Decoder
            self.dec3 = ResBlock(base_channels * 8, base_channels * 2, time_emb_dim)
            self.dec2 = ResBlock(base_channels * 4, base_channels, time_emb_dim)
            self.dec1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)

            self.final = nn.Conv2d(base_channels, in_channels, 1)

            self.down = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        def forward(self, x, t):
            t_emb = self.time_mlp(t)

            # Encoder
            e1 = self.enc1(x, t_emb)
            e2 = self.enc2(self.down(e1), t_emb)
            e3 = self.enc3(self.down(e2), t_emb)

            # Middle
            m = self.mid(self.down(e3), t_emb)

            # Decoder with skip connections
            d3 = self.dec3(torch.cat([self.up(m), e3], dim=1), t_emb)
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), t_emb)
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)

            return self.final(d1)


    class DDPM:
        """DDPM noise schedule and sampling."""

        def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
            self.timesteps = timesteps
            self.device = device

            # Linear noise schedule
            self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
            self.alphas = 1.0 - self.betas
            self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

        def q_sample(self, x_0, t, noise=None):
            """Forward diffusion process - add noise."""
            if noise is None:
                noise = torch.randn_like(x_0)

            alpha_t = self.alpha_cumprod[t][:, None, None, None]
            return torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise

        def p_losses(self, model, x_0):
            """Calculate training loss."""
            batch_size = x_0.shape[0]
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
            noise_pred = model(x_t, t.float())
            return F.mse_loss(noise_pred, noise)


    def main():
        batch_size = 64
        epochs = 100
        lr = 1e-4
        timesteps = 1000

        with flexium.auto.run():
            # Data
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            dataset = datasets.MNIST(
                "./data", train=True, download=True, transform=transform
            )
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )

            # Model
            model = UNet(in_channels=1, base_channels=64).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs * len(dataloader)
            )

            # Diffusion
            ddpm = DDPM(timesteps=timesteps, device="cuda")

            for epoch in range(epochs):
                total_loss = 0
                for i, (images, _) in enumerate(dataloader):
                    images = images.cuda()

                    optimizer.zero_grad()
                    loss = ddpm.p_losses(model, images)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                    if i % 100 == 0:
                        print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                              f"Loss: {loss.item():.4f}")

                avg_loss = total_loss / len(dataloader)
                print(f">>> Epoch {epoch} | Avg Loss: {avg_loss:.4f}")


    if __name__ == "__main__":
        main()
    ```

---

### Transformer Training (GPT-style)

Training a GPT-style language model:

??? example "GPT-style Transformer Training Script"

    ```python
    #!/usr/bin/env python
    """GPT-style Transformer training with flexium.

    A decoder-only transformer language model.
    Demonstrates handling of large sequence models,
    attention mechanisms, and causal masking.
    """

    import math
    import flexium.auto
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset


    class MultiHeadAttention(nn.Module):
        """Multi-head self-attention with causal masking."""

        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            assert d_model % n_heads == 0

            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads

            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            batch_size, seq_len, _ = x.shape

            # Linear projections
            q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Causal mask
            if mask is None:
                mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))

            # Softmax and output
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

            # Reshape and project
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.w_o(out)


    class FeedForward(nn.Module):
        """Position-wise feed-forward network."""

        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            return self.linear2(self.dropout(F.gelu(self.linear1(x))))


    class TransformerBlock(nn.Module):
        """Transformer decoder block."""

        def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.ff = FeedForward(d_model, d_ff, dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            x = x + self.dropout(self.attn(self.ln1(x), mask))
            x = x + self.dropout(self.ff(self.ln2(x)))
            return x


    class GPT(nn.Module):
        """GPT-style decoder-only transformer."""

        def __init__(
            self,
            vocab_size,
            d_model=512,
            n_heads=8,
            n_layers=6,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1,
        ):
            super().__init__()

            self.d_model = d_model
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_seq_len, d_model)

            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            # Weight tying
            self.head.weight = self.token_emb.weight

            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, x):
            batch_size, seq_len = x.shape

            # Embeddings
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = self.token_emb(x) + self.pos_emb(positions)

            # Transformer blocks
            for block in self.blocks:
                x = block(x)

            # Output
            x = self.ln_f(x)
            logits = self.head(x)

            return logits


    class TextDataset(Dataset):
        """Simple character-level text dataset."""

        def __init__(self, text, seq_len):
            self.seq_len = seq_len
            self.chars = sorted(set(text))
            self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
            self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
            self.data = torch.tensor([self.char_to_idx[c] for c in text], dtype=torch.long)

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.seq_len]
            y = self.data[idx + 1:idx + self.seq_len + 1]
            return x, y

        @property
        def vocab_size(self):
            return len(self.chars)


    def main():
        # Hyperparameters
        batch_size = 32
        seq_len = 128
        epochs = 50
        lr = 3e-4
        d_model = 256
        n_heads = 4
        n_layers = 4

        with flexium.auto.run():
            # Load text data (using Shakespeare as example)
            # Download: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
            try:
                with open("data/shakespeare.txt", "r") as f:
                    text = f.read()
            except FileNotFoundError:
                # Generate dummy data if file not found
                print("Shakespeare data not found, using dummy data")
                text = "Hello world! " * 10000

            dataset = TextDataset(text, seq_len)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

            # Model
            model = GPT(
                vocab_size=dataset.vocab_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                max_seq_len=seq_len,
            ).cuda()

            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs * len(dataloader)
            )

            for epoch in range(epochs):
                total_loss = 0
                for i, (x, y) in enumerate(dataloader):
                    x, y = x.cuda(), y.cuda()

                    optimizer.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, dataset.vocab_size), y.view(-1))
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                    if i % 100 == 0:
                        print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                              f"Loss: {loss.item():.4f} PPL: {math.exp(loss.item()):.2f}")

                avg_loss = total_loss / len(dataloader)
                print(f">>> Epoch {epoch} | Avg Loss: {avg_loss:.4f} | "
                      f"PPL: {math.exp(avg_loss):.2f}")

                # Generate sample
                if epoch % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        start = torch.tensor([[dataset.char_to_idx["H"]]]).cuda()
                        generated = [start[0, 0].item()]
                        for _ in range(100):
                            logits = model(start)[:, -1, :]
                            probs = F.softmax(logits / 0.8, dim=-1)
                            next_token = torch.multinomial(probs, 1)
                            generated.append(next_token[0, 0].item())
                            start = torch.cat([start, next_token], dim=1)[:, -seq_len:]

                        text = "".join([dataset.idx_to_char[i] for i in generated])
                        print(f"Sample: {text[:200]}")
                    model.train()


    if __name__ == "__main__":
        main()
    ```

---

### Vision Transformer (ViT) Training

Training a Vision Transformer for image classification:

??? example "Vision Transformer (ViT) Training Script"

    ```python
    #!/usr/bin/env python
    """Vision Transformer (ViT) training with flexium.

    A Vision Transformer for CIFAR-10 classification.
    Demonstrates patch embedding, positional encoding,
    and transformer encoder for vision tasks.
    """

    import flexium.auto
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms


    class PatchEmbedding(nn.Module):
        """Convert image into patches and embed them."""

        def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.n_patches = (img_size // patch_size) ** 2

            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )

        def forward(self, x):
            # (B, C, H, W) -> (B, embed_dim, n_patches_h, n_patches_w) -> (B, n_patches, embed_dim)
            x = self.proj(x)
            x = x.flatten(2).transpose(1, 2)
            return x


    class TransformerEncoder(nn.Module):
        """Standard transformer encoder block."""

        def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, dropout=0.1):
            super().__init__()
            self.norm1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.mlp(self.norm2(x))
            return x


    class ViT(nn.Module):
        """Vision Transformer for image classification."""

        def __init__(
            self,
            img_size=32,
            patch_size=4,
            in_channels=3,
            n_classes=10,
            embed_dim=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
        ):
            super().__init__()

            self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
            n_patches = self.patch_embed.n_patches

            # Learnable class token and position embeddings
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
            self.dropout = nn.Dropout(dropout)

            # Transformer encoder
            self.encoder = nn.ModuleList([
                TransformerEncoder(embed_dim, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ])

            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, n_classes)

            # Initialize weights
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        def forward(self, x):
            batch_size = x.shape[0]

            # Patch embedding
            x = self.patch_embed(x)

            # Add class token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Add position embedding
            x = x + self.pos_embed
            x = self.dropout(x)

            # Transformer encoder
            for block in self.encoder:
                x = block(x)

            # Classification head (use class token)
            x = self.norm(x)
            x = x[:, 0]  # Class token
            x = self.head(x)

            return x


    def main():
        batch_size = 128
        epochs = 100
        lr = 3e-4

        with flexium.auto.run():
            # Data augmentation
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            ])

            train_dataset = datasets.CIFAR10(
                "./data", train=True, download=True, transform=train_transform
            )
            test_dataset = datasets.CIFAR10(
                "./data", train=False, transform=test_transform
            )

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            # Model
            model = ViT(
                img_size=32,
                patch_size=4,
                n_classes=10,
                embed_dim=256,
                n_layers=6,
                n_heads=8,
            ).cuda()

            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            best_acc = 0

            for epoch in range(epochs):
                # Training
                model.train()
                total_loss = 0
                correct = 0
                total = 0

                for i, (images, labels) in enumerate(train_loader):
                    images, labels = images.cuda(), labels.cuda()

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

                    if i % 100 == 0:
                        print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(train_loader)}] "
                              f"Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}%")

                scheduler.step()

                # Evaluation
                model.eval()
                test_correct = 0
                test_total = 0

                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.cuda(), labels.cuda()
                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        test_total += labels.size(0)
                        test_correct += predicted.eq(labels).sum().item()

                test_acc = 100. * test_correct / test_total
                if test_acc > best_acc:
                    best_acc = test_acc

                print(f">>> Epoch {epoch} | Train Acc: {100.*correct/total:.2f}% | "
                      f"Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")


    if __name__ == "__main__":
        main()
    ```

---

## Running the Examples

```bash
# Start orchestrator
flexium-ctl server --dashboard

# Run MNIST example
python examples/simple/mnist_train_auto.py

# Run with custom settings
python examples/simple/mnist_train_auto.py --orchestrator localhost:80 --epochs 5

# Run without flexium (baseline)
python examples/simple/mnist_train_auto.py --disabled
```

---

## Zero-Residue Migration

Flexium's key feature is zero-residue GPU migration - when your training moves from one GPU to another, the source GPU has **0 MB** of memory left behind.

### How It Works

Flexium uses proprietary migration technology that operates at a lower level than PyTorch's memory management, ensuring complete GPU memory release.

1. **Checkpoint**: Training state is safely preserved using Flexium's migration engine
2. **Release**: Source GPU resources are completely freed
3. **Restore**: State is restored on target GPU
4. **Continue**: Training continues seamlessly

No API changes required - zero-residue migration is automatic.

For more details, see [Zero-Residue Migration](features/zero-residue-migration.md).

---

## See Also

- [API Reference](api.md)
- [Architecture Overview](ARCHITECTURE.md)
- [Zero-Residue Migration](features/zero-residue-migration.md)
- [Troubleshooting](troubleshooting.md)
