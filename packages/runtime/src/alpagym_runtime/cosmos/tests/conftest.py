# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for cosmos rollout-backend tests."""

import importlib
import importlib.util
import sys
import types
from typing import Any, cast

import pytest
from alpagym_runtime.alpasim.tests.test_proto_conversion import install_alpasim_grpc_stubs


def _install_cosmos_stubs() -> None:
    """Install minimal Cosmos-RL stubs needed by alpagym runtime unit tests."""
    if "grpc" not in sys.modules and importlib.util.find_spec("grpc") is None:
        grpc_module: Any = types.ModuleType("grpc")

        class _FakeGrpcServer:
            """Tiny stand-in for grpc.Server."""

            def __init__(self, *args: object, **kwargs: object) -> None:
                """Accept server construction arguments."""
                del args, kwargs
                self.servicer: object | None = None

            def add_insecure_port(self, address: str) -> int:
                """Return a deterministic bound port."""
                del address
                return 50051

            def start(self) -> None:
                """Accept server start."""

            def stop(self, grace: float) -> None:
                """Accept server stop."""
                del grace

        def server(*args: object, **kwargs: object) -> _FakeGrpcServer:
            """Build a fake gRPC server."""
            return _FakeGrpcServer(*args, **kwargs)

        grpc_module.server = server
        sys.modules["grpc"] = grpc_module

    transformers_module = sys.modules.get("transformers")
    transformers_is_available = transformers_module is not None and not getattr(
        transformers_module, "__alpagym_test_stub__", False
    )
    if transformers_module is None:
        transformers_is_available = importlib.util.find_spec("transformers") is not None

    if not transformers_is_available:
        transformers: Any = types.ModuleType("transformers")
        transformers.__alpagym_test_stub__ = True

        class AutoConfig:
            """Tiny stand-in for Transformers AutoConfig registration."""

            @classmethod
            def register(cls, *args: object, **kwargs: object) -> None:
                """Accept model config registration."""
                del args, kwargs

        class AutoTokenizer:
            """Tiny stand-in for Transformers AutoTokenizer registration."""

            @classmethod
            def register(cls, *args: object, **kwargs: object) -> None:
                """Accept tokenizer registration."""
                del args, kwargs

            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> object:
                """Default implementation rejects calls; tests must monkeypatch."""
                del args, kwargs
                raise NotImplementedError(
                    "transformers.AutoTokenizer.from_pretrained must be monkeypatched in tests"
                )

        class PretrainedConfig:
            """Tiny stand-in for the Transformers config base class."""

        class PreTrainedTokenizer:
            """Tiny stand-in for the Transformers tokenizer base class."""

            def __init__(self, **kwargs: object) -> None:
                """Accept tokenizer construction kwargs."""
                del kwargs

        class AutoModel:
            """Tiny stand-in for Transformers AutoModel; tests monkeypatch `from_pretrained`."""

            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> object:
                """Default implementation rejects calls; tests must monkeypatch."""
                del args, kwargs
                raise NotImplementedError(
                    "transformers.AutoModel.from_pretrained must be monkeypatched in tests"
                )

        class AutoProcessor:
            """Tiny stand-in for Transformers AutoProcessor.
            Tests monkeypatch `from_pretrained`.
            """

            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> object:
                """Default implementation rejects calls; tests must monkeypatch."""
                del args, kwargs
                raise NotImplementedError(
                    "transformers.AutoProcessor.from_pretrained must be monkeypatched in tests"
                )

        class PreTrainedModel:
            """Tiny stand-in for the Transformers model base class."""

        transformers.AutoConfig = AutoConfig
        transformers.AutoTokenizer = AutoTokenizer
        transformers.AutoModel = AutoModel
        transformers.AutoProcessor = AutoProcessor
        transformers.PretrainedConfig = PretrainedConfig
        transformers.PreTrainedTokenizer = PreTrainedTokenizer
        transformers.PreTrainedModel = PreTrainedModel
        sys.modules["transformers"] = transformers

    _install_alpamayo_r1_recipe_stubs()

    cosmos_rl: Any = types.ModuleType("cosmos_rl")
    sys.modules["cosmos_rl"] = cosmos_rl

    packer_base: Any = types.ModuleType("cosmos_rl.dispatcher.data.packer.base")

    class DataPacker:
        """Tiny stand-in for the Cosmos DataPacker base class."""

        def setup(self, config: object, *args: object, **kwargs: object) -> None:
            """Store config like the real Cosmos DataPacker base class."""
            del args, kwargs
            self.config = config

    packer_base.DataPacker = DataPacker
    packer_base.BaseDataPacker = DataPacker
    sys.modules["cosmos_rl.dispatcher"] = types.ModuleType("cosmos_rl.dispatcher")
    dispatcher_status: Any = types.ModuleType("cosmos_rl.dispatcher.status")

    class PolicyStatusManager:
        """Tiny stand-in for the Cosmos PolicyStatusManager."""

    dispatcher_status.PolicyStatusManager = PolicyStatusManager
    sys.modules["cosmos_rl.dispatcher.status"] = dispatcher_status
    sys.modules["cosmos_rl.dispatcher.data"] = types.ModuleType("cosmos_rl.dispatcher.data")
    sys.modules["cosmos_rl.dispatcher.data.packer"] = types.ModuleType(
        "cosmos_rl.dispatcher.data.packer"
    )
    sys.modules["cosmos_rl.dispatcher.data.packer.base"] = packer_base
    dispatcher_schema: Any = types.ModuleType("cosmos_rl.dispatcher.data.schema")

    class Rollout:
        """Tiny stand-in for the Cosmos Rollout schema."""

    class RLPayload:
        """Tiny stand-in for the Cosmos RLPayload schema."""

    dispatcher_schema.Rollout = Rollout
    dispatcher_schema.RLPayload = RLPayload
    sys.modules["cosmos_rl.dispatcher.data.schema"] = dispatcher_schema

    rollout_base: Any = types.ModuleType("cosmos_rl.rollout.rollout_base")

    class RolloutBase:
        """Tiny stand-in for the Cosmos RolloutBase class."""

        def __init__(self, config: object, parallel_dims: object = None, device=None):
            """Store config and call the subclass post-init hook."""
            self.config = config
            self.parallel_dims = parallel_dims
            self.device = device
            self._engine_initialized = False
            cast(Any, self).post_init_hook()

        def is_engine_initialized(self) -> bool:
            """Return whether the rollout engine has been initialized."""
            return self._engine_initialized

        def get_quantized_tensors(self, weight_mapper: object) -> dict[str, object]:
            """Return no quantized tensors for the test rollout stub."""
            del weight_mapper
            return {}

        def model_param_map(self, weight_mapper: object) -> dict[str, object]:
            """Mirror the Cosmos parameter-map contract used during weight sync."""
            if not self._engine_initialized:
                raise RuntimeError(
                    "[Rollout] Engine is not initialized, please call init_engine first."
                )
            if self._model_param_map:
                return self._model_param_map

            param_map = {}
            state_dict = cast(Any, self).get_underlying_model().state_dict()
            for name, param in state_dict.items():
                mapped_name = cast(Any, weight_mapper).rollout_map_local_key_to_hf_key(name)
                param_map[mapped_name] = param
            param_map.update(self.get_quantized_tensors(weight_mapper))
            self._model_param_map = param_map
            return self._model_param_map

    class RolloutRegistry:
        """Tiny stand-in for the Cosmos rollout registry."""

        _registry: dict[str, type] = {}

        @classmethod
        def register(cls, rollout_type: str, allow_override: bool = False):
            """Register a rollout class under a Cosmos backend name."""
            del allow_override

            def decorator(rollout_cls: type) -> type:
                """Store and return the decorated rollout class."""
                cls._registry[rollout_type] = rollout_cls
                return rollout_cls

            return decorator

        @classmethod
        def get_rollout_cls(cls, rollout_type: str) -> type:
            """Return a registered rollout class."""
            return cls._registry[rollout_type]

    rollout_base.RolloutBase = RolloutBase
    rollout_base.RolloutRegistry = RolloutRegistry
    sys.modules["cosmos_rl.rollout"] = types.ModuleType("cosmos_rl.rollout")
    sys.modules["cosmos_rl.rollout.rollout_base"] = rollout_base

    rollout_schema: Any = types.ModuleType("cosmos_rl.rollout.schema")

    class RolloutResult:
        """Tiny stand-in for the Cosmos RolloutResult model."""

        def __init__(self, completions: list[str]):
            """Store completion handles returned by rollout generation."""
            self.completions = completions

    rollout_schema.RolloutResult = RolloutResult
    sys.modules["cosmos_rl.rollout.schema"] = rollout_schema

    trainer_base: Any = types.ModuleType("cosmos_rl.policy.trainer.base")

    class Trainer:
        """Tiny stand-in for the Cosmos Trainer base class."""

        def __init__(self, config: object, parallel_dims: object = None, **kwargs):
            """Store trainer construction inputs."""
            del kwargs
            self.config = config
            self.parallel_dims = parallel_dims

    class TrainerRegistry:
        """Tiny stand-in for the Cosmos trainer registry."""

        _registry: dict[str, type] = {}

        @classmethod
        def register(cls, trainer_type: str, allow_override: bool = False):
            """Register a trainer class under a Cosmos trainer type."""
            del allow_override

            def decorator(trainer_cls: type) -> type:
                """Store and return the decorated trainer class."""
                cls._registry[trainer_type] = trainer_cls
                return trainer_cls

            return decorator

        @classmethod
        def get_trainer_cls(cls, trainer_type: str) -> type:
            """Return a registered trainer class."""
            return cls._registry[trainer_type]

    trainer_base.Trainer = Trainer
    trainer_base.TrainerRegistry = TrainerRegistry
    sys.modules["cosmos_rl.policy"] = types.ModuleType("cosmos_rl.policy")
    policy_config: Any = types.ModuleType("cosmos_rl.policy.config")

    class Config:
        """Tiny stand-in for the Cosmos policy config object."""

    class GrpoConfig:
        """Tiny stand-in for the Cosmos GRPO config object."""

    policy_config.Config = Config
    policy_config.GrpoConfig = GrpoConfig
    sys.modules["cosmos_rl.policy.config"] = policy_config
    sys.modules["cosmos_rl.policy.trainer"] = types.ModuleType("cosmos_rl.policy.trainer")
    sys.modules["cosmos_rl.policy.trainer.base"] = trainer_base

    llm_trainer: Any = types.ModuleType("cosmos_rl.policy.trainer.llm_trainer")
    grpo_trainer: Any = types.ModuleType("cosmos_rl.policy.trainer.llm_trainer.grpo_trainer")

    class GRPOTrainer(Trainer):
        """Tiny stand-in for Cosmos-RL GRPOTrainer."""

    grpo_trainer.GRPOTrainer = GRPOTrainer
    sys.modules["cosmos_rl.policy.trainer.llm_trainer"] = llm_trainer
    sys.modules["cosmos_rl.policy.trainer.llm_trainer.grpo_trainer"] = grpo_trainer

    model_base: Any = types.ModuleType("cosmos_rl.policy.model.base")

    class WeightMapper:
        """Tiny stand-in for the Cosmos WeightMapper base class."""

        def __init__(self, *args: object, **kwargs: object):
            """Accept Cosmos weight-mapper construction arguments."""
            del args, kwargs

        def rollout_map_local_key_to_hf_key(self, param_name: str) -> str:
            """Return the unchanged rollout parameter name."""
            return param_name

    class ModelRegistry:
        """Tiny stand-in for the Cosmos model registry."""

        _registry: dict[str, type] = {}

        @classmethod
        def register(cls, weight_mapper_cls: type, allow_override: bool = False):
            """Register a model class for every supported model type."""
            del allow_override

            def decorator(model_cls: type) -> type:
                """Store and return the decorated model class."""
                for model_type in getattr(model_cls, "supported_model_types")():
                    cls._registry[model_type] = model_cls
                return model_cls

            del weight_mapper_cls
            return decorator

    model_base.ModelRegistry = ModelRegistry
    model_base.WeightMapper = WeightMapper
    sys.modules["cosmos_rl.policy.model"] = types.ModuleType("cosmos_rl.policy.model")
    sys.modules["cosmos_rl.policy.model.base"] = model_base

    worker_entry: Any = types.ModuleType("cosmos_rl.launcher.worker_entry")
    worker_entry.main = lambda **kwargs: kwargs
    sys.modules["cosmos_rl.launcher"] = types.ModuleType("cosmos_rl.launcher")
    sys.modules["cosmos_rl.launcher.worker_entry"] = worker_entry

    distributed: Any = types.ModuleType("cosmos_rl.utils.distributed")

    class HighAvailabilitylNccl:
        """Tiny stand-in for Cosmos high-availability NCCL helper."""

    parallelism: Any = types.ModuleType("cosmos_rl.utils.parallelism")

    class ParallelDims:
        """Tiny stand-in for Cosmos parallelism dimensions."""

    util: Any = types.ModuleType("cosmos_rl.utils.util")
    util.setup_tokenizer = lambda path: object()
    distributed.HighAvailabilitylNccl = HighAvailabilitylNccl
    parallelism.ParallelDims = ParallelDims
    # payload_transport: stub the cosmos NCCL completion prefix + prefetch mixin
    # so importing alpagym's transport/nccl modules (which reuse these) succeeds
    # even when this session-global cosmos_rl stub shadows the real package. The
    # stubbed prefix must match the real "nccl:" so handle parsing stays
    # consistent across stub and real runs.
    payload_transport: Any = types.ModuleType("cosmos_rl.utils.payload_transport")
    payload_transport_nccl: Any = types.ModuleType("cosmos_rl.utils.payload_transport.nccl")
    payload_transport_nccl.NCCL_COMPLETION_PREFIX = "nccl:"
    payload_transport_nccl.build_nccl_prefix = lambda *, experiment_name, job_id: (
        f"{experiment_name}:{job_id}"
    )
    payload_transport_nccl.build_rollout_prefix = lambda prefix, rollout_idx: (
        f"{prefix}:rollout:{rollout_idx}"
    )
    payload_transport_nccl.build_cleanup_channel = lambda prefix: f"{prefix}:cleanup"
    payload_transport_prefetch: Any = types.ModuleType(
        "cosmos_rl.utils.payload_transport.prefetch_mixin"
    )

    class PrefetchDataPackerMixin:
        """Tiny stand-in for the Cosmos prefetch/double-buffer packer mixin."""

    payload_transport_prefetch.PrefetchDataPackerMixin = PrefetchDataPackerMixin
    payload_transport.nccl = payload_transport_nccl
    payload_transport.prefetch_mixin = payload_transport_prefetch

    utils_pkg: Any = types.ModuleType("cosmos_rl.utils")
    utils_pkg.distributed = distributed
    utils_pkg.parallelism = parallelism
    utils_pkg.util = util
    utils_pkg.payload_transport = payload_transport
    sys.modules["cosmos_rl.utils"] = utils_pkg
    sys.modules["cosmos_rl.utils.distributed"] = distributed
    sys.modules["cosmos_rl.utils.parallelism"] = parallelism
    sys.modules["cosmos_rl.utils.util"] = util
    sys.modules["cosmos_rl.utils.payload_transport"] = payload_transport
    sys.modules["cosmos_rl.utils.payload_transport.nccl"] = payload_transport_nccl
    sys.modules["cosmos_rl.utils.payload_transport.prefetch_mixin"] = payload_transport_prefetch

    for module_name in (
        "alpagym_runtime.cosmos.rollout_backend",
        "alpagym_runtime.cosmos.trainer",
        "alpagym_runtime.cosmos.packer",
    ):
        sys.modules.pop(module_name, None)


def _install_alpamayo_r1_recipe_stubs() -> None:
    """Install R1 recipe stubs only when the real package is unavailable."""
    model_module_name = "alpamayo1_x_rl.models.expert_model.model"
    if model_module_name in sys.modules:
        return
    try:
        recipe_model_spec = importlib.util.find_spec(model_module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        recipe_model_spec = None
    if recipe_model_spec is not None:
        return

    alpamayo1_x_rl: Any = types.ModuleType("alpamayo1_x_rl")
    alpamayo1_x_rl_models: Any = types.ModuleType("alpamayo1_x_rl.models")
    alpamayo1_x_rl_expert_model: Any = types.ModuleType("alpamayo1_x_rl.models.expert_model")
    alpamayo1_x_rl_expert_model_model: Any = types.ModuleType(model_module_name)
    alpamayo1_x_rl_cosmos_wrapper: Any = types.ModuleType(
        "alpamayo1_x_rl.models.expert_model.cosmos_wrapper"
    )

    class ExpertModelRL:
        """Tiny stand-in for the R1 recipe model; tests monkeypatch `from_pretrained`."""

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> object:
            """Default implementation rejects calls; tests must monkeypatch."""
            del args, kwargs
            raise NotImplementedError(
                "ExpertModelRL.from_pretrained must be monkeypatched in tests"
            )

    class ExpertModelCosmos:
        """Tiny stand-in for the R1 Cosmos wrapper."""

        def forward(self, *args: object, **kwargs: object) -> object:
            """Default implementation rejects calls; tests must monkeypatch."""
            del args, kwargs
            raise NotImplementedError("ExpertModelCosmos.forward must be monkeypatched in tests")

    alpamayo1_x_rl_expert_model_model.ExpertModelRL = ExpertModelRL
    alpamayo1_x_rl_cosmos_wrapper.ExpertModelCosmos = ExpertModelCosmos
    alpamayo1_x_rl_expert_model.model = alpamayo1_x_rl_expert_model_model
    alpamayo1_x_rl_expert_model.cosmos_wrapper = alpamayo1_x_rl_cosmos_wrapper
    alpamayo1_x_rl_models.expert_model = alpamayo1_x_rl_expert_model
    alpamayo1_x_rl.models = alpamayo1_x_rl_models
    sys.modules["alpamayo1_x_rl"] = alpamayo1_x_rl
    sys.modules["alpamayo1_x_rl.models"] = alpamayo1_x_rl_models
    sys.modules["alpamayo1_x_rl.models.expert_model"] = alpamayo1_x_rl_expert_model
    sys.modules[model_module_name] = alpamayo1_x_rl_expert_model_model
    sys.modules["alpamayo1_x_rl.models.expert_model.cosmos_wrapper"] = alpamayo1_x_rl_cosmos_wrapper


@pytest.fixture
def cosmos_stubs() -> None:
    """Install Cosmos-RL, transformers, and alpasim_grpc stubs."""
    _install_cosmos_stubs()
    install_alpasim_grpc_stubs()
    importlib.invalidate_caches()


_install_cosmos_stubs()
install_alpasim_grpc_stubs()
importlib.invalidate_caches()
