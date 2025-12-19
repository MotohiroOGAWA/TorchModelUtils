# TorchModelUtils

Hello! Welcome to TorchModelUtils - a collection of utilities for PyTorch model development and training.

## Overview

TorchModelUtils provides essential utilities to streamline PyTorch model development, training, and checkpoint management. It offers a robust framework for managing model configurations, checkpoints with branching support, early stopping, and custom datasets.

## Features

### 🏗️ ModelBase
A base class for PyTorch models with automatic configuration management:
- Automatically records constructor arguments
- Easy model serialization and deserialization
- Configuration export/import via YAML
- Built-in gradient norm logging

### 📦 CheckPointManager
Advanced checkpoint management system with unique features:
- **Branching support**: Create and manage multiple training branches
- **Top-K model tracking**: Automatically keep best performing models
- **Metrics logging**: Track training/validation metrics across checkpoints
- **TensorBoard integration**: Built-in support for visualization
- **Checkpoint lineage**: Trace model evolution through training history

### ⏰ EarlyStopping
Intelligent early stopping with windowed comparison:
- Configurable patience and window size
- Minimum delta threshold for improvements
- Counter reset mechanism for flexible training

### 📊 BsDataset
Dictionary-based dataset for flexible data handling:
- Key-value tensor organization
- Compatible with standard PyTorch DataLoader
- Easy indexing support

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/MotohiroOGAWA/TorchModelUtils.git
cd TorchModelUtils
pip install -e .
```

## Quick Start

### Using ModelBase

```python
from modeling import ModelBase
import torch.nn as nn

class MyModel(ModelBase):
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super(MyModel, self).__init__(
            ignore_config_keys=[],
            **{k: v for k, v in locals().items() if k != 'self'}
        )
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)

# Create model
model = MyModel(input_dim=10, hidden_dim=20, output_dim=2)

# Save configuration
config = model.get_params()

# Recreate from config
new_model = MyModel.from_params(config)
```

### Using CheckPointManager

```python
from modeling import CheckPointManager

# Initialize checkpoint manager
ckpt_manager = CheckPointManager(ckpt_root_dir='./experiments')

# Start a new training branch
root_node = ckpt_manager.checkout_base()

# Save checkpoint
ckpt_node = ckpt_manager.checkout_new_ckpt()
ckpt_node.save_model(
    model=model,
    epoch=10,
    iter=1000,
    optimizer=optimizer
)

# Update metrics
ckpt_manager.initialize_metrics(columns=['epoch', 'loss', 'accuracy'])
ckpt_manager.log_metrics(epoch=10, loss=0.5, accuracy=0.95)
ckpt_manager.flush_metrics()

# Track top-K models
ckpt_manager.update_topk(
    score=0.95,
    epoch=10,
    iter=1000,
    model=model,
    topk=5
)

# Save all changes
ckpt_manager.update()
```

### Using EarlyStopping

```python
from training import EarlyStopping

# Initialize early stopping
early_stopping = EarlyStopping(
    patience=10,
    window_size=5,
    min_delta=0.001,
    reset_step=0.5,
    verbose=True
)

# During training loop
for epoch in range(num_epochs):
    val_loss = validate(model, val_loader)
    early_stopping(val_loss)
    
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break
```

### Using BsDataset

```python
from training import BsDataset
import torch

# Create dataset
dataset = BsDataset(
    input=torch.randn(100, 10),
    target=torch.randint(0, 2, (100,))
)

# Use with DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    inputs = batch['input']
    targets = batch['target']
    # Training code here
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- pandas
- PyYAML
- tensorboard

## Author

Motohiro Ogawa, Tokyo University of Agriculture and Technology

## License

MIT License - see [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Motohiro Ogawa, Tokyo University of Agriculture and Technology

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This project provides utilities to make PyTorch model development more efficient and reproducible.
