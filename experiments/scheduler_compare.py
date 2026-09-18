"""Compare static vs continuous batching on the same workload.

    ./.venv/bin/python -m experiments.scheduler_compare

Both policies run the identical engine, runner and requests; only the
Scheduler admission rule differs (static=True/False). The point is to measure
what static batching costs: steps spent and, more concretely, slot-steps that
were paid for but left empty while requests were queued.

Not learning-critical: this is measurement plumbing around your scheduler.
"""

import contextlib
import io

from engine import Engine, Request, Runner, Scheduler

# (id, prompt_len, max_new_tokens) -- everything arrives at step 0.
# The two 14-token requests are what expose static batching: one long request
# keeps the whole batch (and its slots) held hostage.
WORKLOAD = [
    ("r0", 4, 2),
    ("r1", 4, 2),
    ("r2", 4, 14),
    ("r3", 4, 2),
    ("r4", 4, 2),
    ("r5", 4, 14),
]

MAX_RUNNING = 3


def simulate(static, workload, max_running):
    """Run one policy to completion and return its metrics."""
    scheduler = Scheduler(max_running_requests=max_running, static=static)
    engine = Engine(scheduler, Runner())

    for req_id, prompt_len, max_new in workload:
        engine.submit(Request(req_id, list(range(prompt_len)), 0.0, max_new))

    # Runner prints one line per unit of work; we want them out of the way
    # while collecting numbers. The demo script is where you watch them.
    with contextlib.redirect_stdout(io.StringIO()):
        occupancy = []      # (busy slots, wasted slots) per step
        finish_step = {}
        retired_so_far = 0

        while scheduler.has_work():
            engine.step()
            engine.now += 1

            retired_now = len(scheduler.finished) - retired_so_far
            retired_so_far = len(scheduler.finished)
            for req in scheduler.finished:
                finish_step.setdefault(req.id, engine.now)

            # A request that finished during this step still occupied its slot
            # for the whole step, so count it as busy.
            busy = len(scheduler.running) + retired_now
            queued = bool(scheduler.waiting)
            wasted = (max_running - busy) if queued else 0
            occupancy.append((busy, wasted))

    return {
        "total_steps": engine.now,
        "busy_slot_steps": sum(busy for busy, _ in occupancy),
        "wasted_slot_steps": sum(wasted for _, wasted in occupancy),
        "finish_step": finish_step,
        "occupancy": occupancy,
        "scheduler": scheduler,
    }


def occupancy_line(occupancy, max_running):
    """3 chars per step: '#' busy, '.' idle while queued, '-' idle and empty."""
    chars = []
    for busy, wasted in occupancy:
        chars.append("#" * busy)
        chars.append(("." if wasted else "-") * (max_running - busy))
    return "".join(chars)


def main():
    results = {
        "continuous": simulate(False, WORKLOAD, MAX_RUNNING),
        "static": simulate(True, WORKLOAD, MAX_RUNNING),
    }

    print(f"workload: {len(WORKLOAD)} requests, max_running = {MAX_RUNNING}, "
          f"all arriving at step 0")
    for req_id, prompt_len, max_new in WORKLOAD:
        print(f"  {req_id}  prompt={prompt_len:<3} max_new_tokens={max_new}")
    print()

    header = f"{'policy':<12}{'total steps':>13}{'busy slot-steps':>18}{'wasted slot-steps':>20}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<12}{r['total_steps']:>13}{r['busy_slot_steps']:>18}"
              f"{r['wasted_slot_steps']:>20}")
    print()

    print(f"slot occupancy ({MAX_RUNNING} chars per step)")
    print("  # busy   . idle while requests were queued   - idle, nothing queued")
    for name, r in results.items():
        print(f"  {name:<11} {occupancy_line(r['occupancy'], MAX_RUNNING)}")
    print()

    print("finish step per request (all arrive at 0, so this is also latency)")
    print(f"  {'id':<6}{'continuous':>12}{'static':>9}{'added':>8}")
    for req_id, _, _ in WORKLOAD:
        cont = results["continuous"]["finish_step"][req_id]
        stat = results["static"]["finish_step"][req_id]
        print(f"  {req_id:<6}{cont:>12}{stat:>9}{stat - cont:>+8}")
    print()

    # Same correctness check as the demo: the policy must not change results.
    for name, r in results.items():
        sched = r["scheduler"]
        assert not sched.waiting and not sched.running, name
        for req in sched.finished:
            assert len(req.generated_tokens) == req.max_new_tokens, (name, req)
        assert len(sched.finished) == len(WORKLOAD), name
    print("checks passed: both policies produced every token for every request")

    print()
    wasted = results["static"]["wasted_slot_steps"]
    extra = results["static"]["total_steps"] - results["continuous"]["total_steps"]
    print(f"static batching cost {extra} extra steps and wasted {wasted} slot-steps "
          f"that continuous batching put to work.")


if __name__ == "__main__":
    main()
