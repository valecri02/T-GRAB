import importlib
import sys
from pathlib import Path

import torch


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR.parent))

no_memory_module = importlib.import_module("T-GRAB.model.TGB.no_memory")
TGNNoMemory = no_memory_module.TGNNoMemory


def test_no_memory_returns_zeros_and_tracks_last_update():
    node_features = torch.randn(4, 3)
    model = TGNNoMemory(
        num_nodes=4,
        memory_dim=3,
        time_dim=5,
        node_raw_features=node_features,
        init_time=2,
    )

    n_id = torch.tensor([0, 1, 2, 3])
    memory, last_update = model(n_id)
    torch.testing.assert_close(memory, torch.zeros(4, 3))
    torch.testing.assert_close(last_update, torch.full((4,), 2, dtype=torch.long))

    model.update_state(
        src=torch.tensor([0, 1]),
        dst=torch.tensor([1, 2]),
        t=torch.tensor([4, 7]),
        raw_msg=torch.randn(2, 2),
    )
    _, last_update = model(n_id)
    torch.testing.assert_close(last_update, torch.tensor([7, 7, 7, 2]))

    model.reset_state()
    memory, last_update = model(n_id)
    torch.testing.assert_close(memory, torch.zeros(4, 3))
    torch.testing.assert_close(last_update, torch.zeros(4, dtype=torch.long))


def test_no_memory_has_no_recurrent_parameters():
    model = TGNNoMemory(
        num_nodes=4,
        memory_dim=3,
        time_dim=5,
        node_raw_features=torch.randn(4, 3),
    )

    parameter_names = {name for name, _ in model.named_parameters()}
    assert all("memory_updater" not in name for name in parameter_names)
    assert all("msg_" not in name for name in parameter_names)
    assert any("time_enc" in name for name in parameter_names)
