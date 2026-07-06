"""Quick median-of-N microbench wrapper to reduce noise."""
import argparse
import subprocess
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--positions", type=int, default=32)
    p.add_argument("--simulations", type=int, default=400)
    args = p.parse_args()

    samples = []
    for i in range(args.runs):
        out = subprocess.run(
            [sys.executable, "mcts_micro_bench.py",
             "--positions", str(args.positions),
             "--simulations", str(args.simulations)],
            capture_output=True, text=True,
            cwd="c:/Users/angel/Downloads/BarricadeGame",
        )
        # Find sims/sec= in stdout
        for line in out.stdout.splitlines():
            if "sims/sec=" in line:
                val = float(line.split("sims/sec=")[1].strip())
                samples.append(val)
                print(f"run {i+1}: {val:.1f} sims/sec")
                break

    if samples:
        samples.sort()
        median = samples[len(samples) // 2]
        print(f"\nmedian over {len(samples)} runs: {median:.1f} sims/sec")
        print(f"all: {samples}")


if __name__ == "__main__":
    main()