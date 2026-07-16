"""Locomotion subsystem for the G1 safety bench (agent A4).

Primary: ONNX velocity policy (from unitree_g1_ros2_driver branch
vsp/feature/onnx-loco-controller) driven over unitree DDS (rt/lowstate ->
rt/lowcmd), commanded via ZMQ SUB on tcp://127.0.0.1:5556.

Entry points:
    python -m g1_safety_bench.loco.runner          # ONNX policy runner (primary)
    python -m g1_safety_bench.loco.scripted_base   # scripted fallback (no policy)
"""
