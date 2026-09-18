"""Compare static vs continuous batching on the same workload.

    ./.venv/bin/python -m experiments.scheduler_compare

Both policies run the identical engine, runner and requests; only the
Scheduler admission rule differs (static=True/False). The point is to measure
what static batching costs: wall-clock, and more concretely the slot-steps that
were paid for but left empty while arrived requests sat in the queue.

Not learning-critical: this is measurement plumbing around your scheduler.
"""

import contextlib
import io

from engine import Engine, Request, Runner, Scheduler

# (id, prompt_len, max_new_tokens, arrival_time). Staggered arrivals: the two
# long requests land later, so the queue is sometimes empty (nothing to admit,
# which is not waste) and sometimes backed up (which is).
WORKLOAD = [
    ("r0", 4, 2, 0.0),
    ("r1", 4, 2, 0.0),
    ("r2", 4, 14, 0.0),
    ("r3", 4, 2, 3.0),
    ("r4", 4, 2, 4.5),
    ("r5", 4, 14, 6.0),
]

MAX_RUNNING = 3


def simulate(static, workload, max_running):
    """Run one policy to completion and return its metrics."""
    scheduler = Scheduler(max_running_requests=max_running, static=static)
    engine = Engine(scheduler, Runner())

    for req_id, prompt_len, max_new, arrival in workload:
        engine.submit(Request(req_id, list(range(prompt_len)), arrival, max_new))

    # Runner prints one line per unit of work; keep them out of the numbers.
    # The demo script is where you watch them.
    with contextlib.redirect_stdout(io.StringIO()):
        occupancy = []      # (busy slots, wasted slots) per step
        finish_time = {}
        retired_so_far = 0

        while scheduler.has_work():
            step_time = engine.now
            engine.step()

            retired_now = len(scheduler.finished) - retired_so_far
            retired_so_far = len(scheduler.finished)
            for req in scheduler.finished:
                finish_time.setdefault(req.id, engine.now)

            # A request that finished during this step still occupied its slot
            # for the whole step, so count it as busy.
            busy = len(scheduler.running) + retired_now

            # Waste is a slot sitting empty while an *arrived* request is
            # queued. A request that has not arrived yet is not work you could
            # have done, so idle slots while it is still pending are not waste.
            queued = any(r.arrival_time <= step_time for r in scheduler.waiting)
            wasted = (max_running - busy) if queued else 0

            occupancy.append((busy, wasted))

    return {
        "steps": engine.now,
        "makespan": max(finish_time.values()),
        "busy_slot_steps": sum(busy for busy, _ in occupancy),
        "wasted_slot_steps": sum(wasted for _, wasted in occupancy),
        "finish_time": finish_time,
        "occupancy": occupancy,
        "scheduler": scheduler,
    }


def occupancy_line(occupancy, max_running):
    """3 chars per step: '#' busy, '.' idle while work was queued, '-' otherwise."""
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

    print(f"workload: {len(WORKLOAD)} requests, max_running = {MAX_RUNNING}")
    for req_id, prompt_len, max_new, arrival in WORKLOAD:
        print(f"  {req_id}  prompt={prompt_len:<3} max_new_tokens={max_new:<3} "
              f"arrives at t={arrival}")
    print()

    header = (f"{'policy':<12}{'steps':>7}{'makespan':>10}{'busy slot-steps':>18}"
              f"{'wasted slot-steps':>20}")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<12}{r['steps']:>7}{r['makespan']:>10}"
              f"{r['busy_slot_steps']:>18}{r['wasted_slot_steps']:>20}")
    print()

    print(f"slot occupancy ({MAX_RUNNING} chars per step, one step per time unit)")
    print("  # busy   . idle while requests were queued   - idle, nothing queued")
    for name, r in results.items():
        print(f"  {name:<11} {occupancy_line(r['occupancy'], MAX_RUNNING)}")
    print()

    print("finish time and latency per request")
    print(f"  {'id':<6}{'arrive':>8}{'cont@':>8}{'static@':>9}"
          f"{'cont lat':>10}{'static lat':>12}")
    for req_id, _, _, arrival in WORKLOAD:
        cont = results["continuous"]["finish_time"][req_id]
        stat = results["static"]["finish_time"][req_id]
        print(f"  {req_id:<6}{arrival:>8}{cont:>8}{stat:>9}"
              f"{cont - arrival:>10}{stat - arrival:>12}")
    print()

    # Same correctness check as the demo: policy must not change what work was
    # done, only when it happened.
    for name, r in results.items():
        sched = r["scheduler"]
        assert not sched.waiting and not sched.running, name
        for req in sched.finished:
            assert len(req.generated_tokens) == req.max_new_tokens, (name, req)
        assert len(sched.finished) == len(WORKLOAD), name
    print("checks passed: both policies produced every token for every request")

    print()
    wasted = results["static"]["wasted_slot_steps"]
    extra = results["static"]["makespan"] - results["continuous"]["makespan"]
    print(f"static batching finished {extra} time units later and wasted "
          f"{wasted} slot-steps.")


if __name__ == "__main__":
    main()
