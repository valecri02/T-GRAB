import sys
from importlib import import_module
from typing import Any, Dict, Iterable, List

import hydra
from omegaconf import DictConfig, OmegaConf

from .trainer import Trainer


def _find_trainer_class(module_name: str):
    module = import_module(module_name)
    target = None

    for _, obj in vars(module).items():
        if isinstance(obj, type) and issubclass(obj, Trainer) and obj.__module__ == module_name:
            target = obj

    if target is None:
        raise ValueError(f"No Trainer subclass found in module {module_name}.")
    return target


def _module_candidates(trainer_module: str) -> Iterable[str]:
    yield f"T-GRAB.train.{trainer_module}"
    yield f"train.{trainer_module}"


def _load_trainer_class(trainer_module: str):
    errors = []
    for module_name in _module_candidates(trainer_module):
        try:
            return _find_trainer_class(module_name)
        except ModuleNotFoundError as exc:
            errors.append(f"{module_name}: {exc}")
    raise ModuleNotFoundError(
        f"Could not import trainer module '{trainer_module}'. Tried:\n"
        + "\n".join(errors)
    )


def _as_cli_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def _cfg_to_argv(cfg: DictConfig) -> List[str]:
    args: Dict[str, Any] = OmegaConf.to_container(cfg.args, resolve=True)
    argv = ["hydra_multirun"]

    for key, value in args.items():
        if value is None or value is False:
            continue

        cli_key = "--" + key.replace("_", "-")
        if value is True:
            argv.append(cli_key)
        else:
            argv.append(f"{cli_key}={_as_cli_value(value)}")

    return argv


@hydra.main(
    config_path="../config/gnn",
    config_name="experiment_config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    trainer_cls = _load_trainer_class(cfg.trainer)
    argv = _cfg_to_argv(cfg)

    previous_argv = sys.argv[:]
    try:
        sys.argv = argv
        trainer = trainer_cls()
        trainer.run()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
