import subprocess
import time
import datetime

LOG = "/home/thelegendiv/finn/notebooks/enet/finn_deployment_outputs/stitched_ip_partitioned_quantEnet_s19_double_mid_int8_largefifo_rtlsim_20260815_113231/build_dataflow.log"
STATUS_LOG = "/home/thelegendiv/finn/notebooks/enet/_partitioned_build_poll_status.log"
STDOUT_LOG = "/tmp/finn_partitioned_build.log"

def poll_once():
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    lines = []
    lines.append(f"=== poll @ {ts} ===")
    try:
        out = subprocess.run(
            ["bash", "-c", "ps aux | grep -E 'python3|vivado|vitis_hls' | grep -v grep"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        lines.append("processes:\n" + (out if out else "(none running)"))
    except Exception as e:
        lines.append(f"process check failed: {e}")

    try:
        with open(LOG) as f:
            tail = f.readlines()[-15:]
        lines.append("build_dataflow.log tail:\n" + "".join(tail))
    except Exception as e:
        lines.append(f"build_dataflow.log read failed: {e}")

    try:
        out = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10).stdout
        lines.append("memory:\n" + out)
    except Exception as e:
        lines.append(f"free -h failed: {e}")

    entry = "\n".join(lines) + "\n\n"
    with open(STATUS_LOG, "a") as f:
        f.write(entry)
    print(entry)

if __name__ == "__main__":
    # Poll every 30 min. Exits (and thus notifies whoever is watching this
    # terminal) if the build process disappears (finished, crashed, or
    # killed) or after a generous 24h safety cap -- whichever first.
    max_iters = 48  # 48 * 30min = 24h safety cap
    for i in range(max_iters):
        poll_once()
        # check if the build's python driver is still alive (matches either
        # the original launcher or a resume-script relaunch)
        alive = subprocess.run(
            ["bash", "-c", "pgrep -f 'finn_enet_ip_build_partitioned.py|finn_enet_ip_resume_partitioned.py'"],
            capture_output=True, text=True,
        ).stdout.strip()
        if not alive:
            print("Build process no longer running -- stopping poll loop.")
            break
        time.sleep(1800)
    print("Poll loop finished.")
