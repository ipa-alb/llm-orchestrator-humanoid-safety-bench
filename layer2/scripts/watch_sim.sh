#!/usr/bin/env bash
# One-command interactive sim demo: MuJoCo viewer (X11 from container) + ONNX
# walking policy + scripted S2/S3 showcase (walk, human approach, slow-to-30%,
# controlled halt, arm move). Ctrl+C or Enter at the end shuts everything down.
#
#   ./scripts/watch_sim.sh            # run the full showcase
#   ./scripts/watch_sim.sh --no-demo  # just viewer + policy, drive it yourself
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMPOSE="docker compose -f docker/compose.yaml"
DEMO=1
[[ "${1:-}" == "--no-demo" ]] && DEMO=0

# ---- preflight -------------------------------------------------------------
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
if [[ -z "${DISPLAY:-}" ]]; then
  echo "No \$DISPLAY — run this from your graphical session (the viewer needs X11)."
  exit 1
fi
xhost +local: >/dev/null 2>&1 || echo "note: 'xhost +local:' failed — if the viewer window doesn't appear, run it manually."

# swarm integration tests may own the stack/ports — don't fight them
if docker ps --format '{{.Names}}' | grep -qE 'g1bench|g1-sim|sim-gui'; then
  echo "Sim containers are already running (probably the agent swarm's integration tests):"
  docker ps --format '  {{.Names}}  ({{.Image}})' | grep -E 'g1bench|g1-sim|sim-gui' || true
  echo "Stop them first ($COMPOSE down) or retry later."
  exit 1
fi

# images (build once if missing; ~10-20 min first time)
docker image inspect g1-base >/dev/null 2>&1 || docker build -f docker/Dockerfile.base -t g1-base .
docker image inspect g1-sim  >/dev/null 2>&1 || docker build -f docker/Dockerfile.sim  -t g1-sim .

# ---- launch ----------------------------------------------------------------
cleanup() {
  echo; echo ">>> shutting down..."
  $COMPOSE --profile gui down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo ">>> starting viewer (sim-gui) + walking policy (loco)..."
$COMPOSE --profile gui up -d sim-gui loco

echo ">>> waiting for the sim to come up..."
sleep 6
if ! docker ps --format '{{.Names}}' | grep -q 'sim-gui'; then
  echo "sim-gui exited early — logs:"
  $COMPOSE --profile gui logs sim-gui | tail -30
  exit 1
fi

if [[ "$DEMO" == "1" ]]; then
  echo ">>> running the scripted showcase (watch the viewer window)..."
  docker run --rm --network host -v "$PWD":/ws -w /ws \
    -e PYTHONPATH=/ws/bench g1-base python3 scripts/demo_drive.py || true
fi

echo
echo ">>> Viewer stays open. Drive manually from another terminal, e.g.:"
echo '    docker run --rm -it --network host -v "$PWD":/ws -w /ws -e PYTHONPATH=/ws/bench g1-base python3'
echo '    >>> import zmq,json; p=zmq.Context().socket(zmq.PUB); p.bind("tcp://127.0.0.1:5556")'
echo '    >>> p.send_string(json.dumps({"type":"cmd_vel","vx":0.3,"vy":0,"wz":0}))'
echo
read -r -p ">>> Press Enter to shut everything down... "
