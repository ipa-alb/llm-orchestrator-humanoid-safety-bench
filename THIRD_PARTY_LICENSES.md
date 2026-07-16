# Third-party licenses

This repository's own code and data are MIT-licensed (see `LICENSE`).
It additionally uses the following third-party components:

## Unitree submodules (BSD 3-Clause)

The submodules under `humanoid_testing/vendor/` are © HangZhou YuShu
TECHNOLOGY CO., LTD. ("Unitree Robotics") and licensed under the
**BSD 3-Clause License** — see the `LICENSE` file inside each submodule:

- `unitree_mujoco` — the G1 robot MJCF model and meshes used by the
  Layer-2 simulation scenes, plus the optional C++ simulator.
- `unitree_sdk2` — C++ SDK (used only to build the optional C++ simulator
  path in the Docker images).
- `unitree_sdk2_python` — Python SDK (DDS data plane of the Layer-2 stack).

## Notable pip/system dependencies (not vendored)

- MuJoCo (Apache-2.0), CycloneDDS (EPL-2.0/EDL-1.0), NumPy (BSD),
  matplotlib (PSF-based), pyzmq (BSD), PyYAML (MIT), onnxruntime (MIT),
  `anthropic` / `openai` SDKs (MIT/Apache-2.0). Installed via
  `pip`/`apt` per the requirements files; their licenses apply as published
  by their maintainers.
