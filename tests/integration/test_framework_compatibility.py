"""Integration tests for framework compatibility.

These tests verify that popular PyTorch frameworks work seamlessly with
flexium.init() - no special integration code needed.

The key insight is that driver-level migration remaps CUDA device indices
at the process level, so any framework using standard PyTorch device
placement (.cuda(), .to(device)) works automatically.

Install test dependencies with:
    pip install -e ".[test-frameworks]"

Run these tests with:
    pytest tests/integration/test_framework_compatibility.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 100, output_size: int = 10):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


class TestPyTorchLightning:
    """Test PyTorch Lightning compatibility."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if lightning not installed."""
        pytest.importorskip("pytorch_lightning")

    def test_lightning_trainer_basic(self):
        """Test basic Lightning Trainer works with flexium.init()."""
        import pytorch_lightning as pl

        import flexium
        flexium.init()

        class LitModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = SimpleModel()

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = nn.functional.mse_loss(y_hat, y)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        # Create simple dataset
        x = torch.randn(100, 100)
        y = torch.randn(100, 10)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        model = LitModel()
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )

        # This should work without any FlexiumCallback
        trainer.fit(model, dataloader)

        assert trainer.current_epoch == 1
        flexium.shutdown()

    def test_lightning_with_validation(self):
        """Test Lightning with validation loop."""
        import pytorch_lightning as pl

        import flexium
        flexium.init()

        class LitModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = SimpleModel()

            def forward(self, x):
                return self.model(x)

            def training_step(self, batch, batch_idx):
                x, y = batch
                loss = nn.functional.mse_loss(self(x), y)
                self.log("train_loss", loss)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                loss = nn.functional.mse_loss(self(x), y)
                self.log("val_loss", loss)

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.001)

        x = torch.randn(100, 100)
        y = torch.randn(100, 10)
        dataset = torch.utils.data.TensorDataset(x, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=10)

        model = LitModel()
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=False,
            logger=False,
        )

        trainer.fit(model, train_loader, val_loader)
        flexium.shutdown()


class TestHuggingFaceTransformers:
    """Test Hugging Face Transformers compatibility."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if transformers not installed."""
        pytest.importorskip("transformers")

    def test_transformers_model_inference(self):
        """Test loading and running a transformers model."""
        from transformers import AutoModel, AutoTokenizer

        import flexium
        flexium.init()

        # Use a tiny model for testing
        model_name = "prajjwal1/bert-tiny"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).cuda()

        # Run inference
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        assert outputs.last_hidden_state is not None
        assert outputs.last_hidden_state.device.type == "cuda"

        flexium.shutdown()

    def test_transformers_training_step(self):
        """Test a training step with transformers model."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        import flexium
        flexium.init()

        model_name = "prajjwal1/bert-tiny"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        # Simulate a training step
        inputs = tokenizer(
            ["Hello, world!", "Goodbye, world!"],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        labels = torch.tensor([0, 1]).cuda()

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        flexium.shutdown()


class TestHuggingFaceAccelerate:
    """Test Hugging Face Accelerate compatibility."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if accelerate not installed."""
        pytest.importorskip("accelerate")

    def test_accelerate_basic(self):
        """Test basic Accelerate usage."""
        from accelerate import Accelerator

        import flexium
        flexium.init()

        accelerator = Accelerator()

        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create dataloader
        x = torch.randn(100, 100)
        y = torch.randn(100, 10)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        # Prepare with accelerator
        model, optimizer, dataloader = accelerator.prepare(
            model, optimizer, dataloader
        )

        # Training step
        for batch in dataloader:
            x, y = batch
            outputs = model(x)
            loss = nn.functional.mse_loss(outputs, y)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            break  # Just one batch for testing

        assert loss.item() > 0
        flexium.shutdown()


class TestTimm:
    """Test timm (PyTorch Image Models) compatibility."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if timm not installed."""
        pytest.importorskip("timm")

    def test_timm_model_inference(self):
        """Test loading and running a timm model."""
        import timm

        import flexium
        flexium.init()

        # Use a small model for testing
        model = timm.create_model("resnet18", pretrained=False, num_classes=10)
        model = model.cuda()

        # Run inference
        x = torch.randn(2, 3, 224, 224).cuda()
        with torch.no_grad():
            outputs = model(x)

        assert outputs.shape == (2, 10)
        assert outputs.device.type == "cuda"

        flexium.shutdown()

    def test_timm_training_step(self):
        """Test a training step with timm model."""
        import timm

        import flexium
        flexium.init()

        model = timm.create_model("resnet18", pretrained=False, num_classes=10)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training step
        x = torch.randn(4, 3, 224, 224).cuda()
        y = torch.randint(0, 10, (4,)).cuda()

        outputs = model(x)
        loss = nn.functional.cross_entropy(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        flexium.shutdown()


class TestPurePyTorch:
    """Test pure PyTorch (baseline) compatibility."""

    def test_basic_training_loop(self):
        """Test standard PyTorch training loop."""
        import flexium
        flexium.init()

        model = SimpleModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for _ in range(5):
            x = torch.randn(32, 100).cuda()
            y = torch.randn(32, 10).cuda()

            outputs = model(x)
            loss = nn.functional.mse_loss(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.item() >= 0
        flexium.shutdown()

    def test_model_to_device_variations(self):
        """Test various ways to move model to device."""
        import flexium
        flexium.init()

        # Test .cuda()
        model1 = SimpleModel().cuda()
        assert next(model1.parameters()).device.type == "cuda"

        # Test .to('cuda')
        model2 = SimpleModel().to('cuda')
        assert next(model2.parameters()).device.type == "cuda"

        # Test .to(device)
        device = torch.device('cuda:0')
        model3 = SimpleModel().to(device)
        assert next(model3.parameters()).device.type == "cuda"

        flexium.shutdown()

    def test_tensor_to_device_variations(self):
        """Test various ways to move tensors to device."""
        import flexium
        flexium.init()

        x = torch.randn(10, 10)

        # Test .cuda()
        x1 = x.cuda()
        assert x1.device.type == "cuda"

        # Test .to('cuda')
        x2 = x.to('cuda')
        assert x2.device.type == "cuda"

        # Test .to(device)
        device = torch.device('cuda:0')
        x3 = x.to(device)
        assert x3.device.type == "cuda"

        flexium.shutdown()
