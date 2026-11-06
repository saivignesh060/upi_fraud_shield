"""
Measure WebSocket stream throughput at several requested tps values.
Excludes the first WARMUP messages from the rate calculation.
"""
import argparse
import asyncio
import time

import websockets


async def measure_one(base_url: str, tps: int, n: int, warmup: int, rows: int) -> dict:
    url = f"{base_url}/stream?tps={tps}&rows={rows}"
    times = []
    async with websockets.connect(url, max_size=8_000_000) as ws:
        for i in range(n):
            await asyncio.wait_for(ws.recv(), timeout=120)
            times.append(time.perf_counter())
    if len(times) <= warmup:
        raise RuntimeError(f"not enough messages: got {len(times)}, warmup={warmup}")
    elapsed = times[-1] - times[warmup]
    counted = len(times) - 1 - warmup
    # intervals between warmup..last => counted messages after first post-warmup
    # Prefer: from after warmup message to last message, count (n - warmup - 1) intervals
    # Rate = (n - warmup) / (t_last - t_warmup) using messages received after warm-up inclusive of last
    counted_msgs = n - warmup
    rate = counted_msgs / elapsed if elapsed > 0 else float("inf")
    return {
        "tps_requested": tps,
        "n": n,
        "warmup": warmup,
        "elapsed_after_warmup_s": round(elapsed, 4),
        "achieved_msg_per_s": round(rate, 2),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="ws://127.0.0.1:8000")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--tps", type=int, nargs="+", default=[10, 20, 50, 100])
    args = p.parse_args()

    results = []
    for tps in args.tps:
        r = await measure_one(args.base, tps, args.n, args.warmup, args.rows)
        results.append(r)
        print(
            f"tps_requested={r['tps_requested']} n={r['n']} warmup={r['warmup']} "
            f"elapsed_after_warmup_s={r['elapsed_after_warmup_s']} "
            f"achieved_msg_per_s={r['achieved_msg_per_s']}"
        )
    print("SUMMARY", {r["tps_requested"]: r["achieved_msg_per_s"] for r in results})


if __name__ == "__main__":
    asyncio.run(main())
