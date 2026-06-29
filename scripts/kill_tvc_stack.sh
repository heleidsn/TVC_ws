#!/usr/bin/env bash
# Kill leftover TVC SITL / ROS2 processes after interrupted launches.
#
# Usage:
#   ./scripts/kill_tvc_stack.sh          # kill and report
#   ./scripts/kill_tvc_stack.sh --check  # only list leftovers, do not kill

set -u

CHECK_ONLY=0
if [[ "${1:-}" == "--check" || "${1:-}" == "-n" ]]; then
    CHECK_ONLY=1
fi

log() {
    printf '%s\n' "$*"
}

list_matches() {
    local pattern="$1"
    pgrep -af "$pattern" 2>/dev/null | grep -v 'cursorsandbox' | grep -v 'kill_tvc_stack' | grep -v 'pgrep -af' || true
}

kill_pattern() {
    local label="$1"
    local pattern="$2"
    local matches
    matches="$(list_matches "$pattern")"
    if [[ -z "$matches" ]]; then
        return 0
    fi

    log "  [$label]"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        log "    $line"
    done <<<"$matches"

    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        return 0
    fi

    pkill -9 -f "$pattern" 2>/dev/null || true
}

kill_name() {
    local label="$1"
    shift
    local name pid matches=""
    for name in "$@"; do
        while read -r pid; do
            [[ -z "$pid" ]] && continue
            matches+=$'\n'"  [$label] $pid $(ps -p "$pid" -o args= 2>/dev/null || true)"
        done < <(pgrep -x "$name" 2>/dev/null || true)
    done

    if [[ -z "$matches" ]]; then
        return 0
    fi

    log "$matches"
    if [[ "$CHECK_ONLY" -eq 1 ]]; then
        return 0
    fi

    for name in "$@"; do
        killall -9 "$name" 2>/dev/null || true
    done
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    log "Checking for leftover TVC / ROS2 / SITL processes..."
else
    log "Stopping leftover TVC / ROS2 / SITL processes..."
fi

kill_pattern "ros2 launch" 'ros2 launch tvc_controller'
kill_pattern "tvc_controller nodes" 'install/tvc_controller/lib/tvc_controller'
kill_pattern "tvc_traj_player" 'tvc_traj_player'
kill_pattern "ros_gz_bridge" 'ros_gz_bridge/parameter_bridge'
kill_pattern "robot_state_publisher" 'robot_state_publisher'
kill_pattern "static_transform_publisher" 'static_transform_publisher'
kill_pattern "rviz2" 'rviz2'
kill_pattern "MicroXRCE agent" 'MicroXRCEAgent|micro-xrce-dds-agent'
kill_pattern "PX4 SITL" 'PX4-TVC-NUS.*px4|build/px4_sitl_default/bin/px4'
kill_pattern "Gazebo" 'gz sim|gz-sim'

kill_name "binary" px4 gz ruby MicroXRCEAgent

if [[ "$CHECK_ONLY" -eq 0 ]]; then
    sleep 1
    if command -v ros2 >/dev/null 2>&1; then
        ros2 daemon stop >/dev/null 2>&1 || true
    fi
    sleep 1
fi

log ""
if [[ "$CHECK_ONLY" -eq 1 ]]; then
    remaining="$( {
        list_matches 'install/tvc_controller|ros_gz_bridge/parameter_bridge|ros2 launch tvc_controller'
        list_matches 'px4|MicroXRCE|gz sim|gz-sim'
    } )"
    if [[ -z "$remaining" ]]; then
        log "No leftover processes found."
    else
        log "Leftover processes still running:"
        log "$remaining"
        exit 1
    fi
else
    remaining="$( {
        list_matches 'install/tvc_controller|ros_gz_bridge/parameter_bridge|ros2 launch tvc_controller'
        list_matches 'px4|MicroXRCE|gz sim|gz-sim'
    } )"
    if [[ -z "$remaining" ]]; then
        log "Done. Stack is clean."
    else
        log "Warning: some processes may still be running:"
        log "$remaining"
        log "Try: pgrep -af 'tvc_controller|ros_gz_bridge' | awk '{print \$1}' | xargs -r kill -9"
        exit 1
    fi
fi
