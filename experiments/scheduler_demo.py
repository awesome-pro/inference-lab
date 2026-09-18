"""Run the scheduler simulation on a set of test requests.

    ./.venv/bin/python -m experiments.scheduler_demo

Prints a per-step trace of the three queues plus a final summary, so you can
watch requests move waiting -> running -> finished and see how the running
slots get reused as requests finish.

Not learning-critical: this is just a driver around Engine/Scheduler/Runner.
"""

from engine import Engine, Request, Runner, Scheduler

# (id, prompt_len, max_new_tokens) -- arrival_time is 0.0 for all of them,
# because the engine does not consume arrival_time yet.
REQUEST_SPECS = [
    ("r0", 8, 3),
    ("r1", 3, 5),
    ("r2", 12, 2),
    ("r3", 5, 4),
    ("r4", 1, 1),
    ("r5", 7, 3),
]

MAX_RUNNING = 3
STATIC = False  # True = hold each batch until it fully drains, no refilling


def make_requests():
    return [
        Request(
            id=req_id,
            prompt_tokens=list(range(prompt_len)),
            arrival_time=0.0,
            max_new_tokens=max_new_tokens,
        )
        for req_id, prompt_len, max_new_tokens in REQUEST_SPECS
    ]


def show_queues(scheduler):
    print("  waiting :", [r.id for r in scheduler.waiting] or "-")
    print("  running :", [r.id for r in scheduler.running] or "-")
    print("  finished:", [r.id for r in scheduler.finished] or "-")


def main():
    scheduler = Scheduler(max_running_requests=MAX_RUNNING, static=STATIC)
    engine = Engine(scheduler, Runner())

    for request in make_requests():
        engine.submit(request)

    print(f"policy = {'static' if STATIC else 'continuous'}")
    print(f"max_running_requests = {MAX_RUNNING}")
    print(f"submitted {len(scheduler.waiting)} requests\n")

    finish_step = {}

    while scheduler.has_work():
        engine.step()
        engine.now += 1

        for req in scheduler.finished:
            finish_step.setdefault(req.id, engine.now)

        print(f"--- after step {engine.now} ---")
        show_queues(scheduler)
        print()

    print("=== summary ===")
    print(f"{'id':<5}{'prompt':>7}{'target':>8}{'generated':>11}{'finished@step':>15}")
    for req_id, prompt_len, max_new_tokens in REQUEST_SPECS:
        req = next(r for r in scheduler.finished if r.id == req_id)
        print(
            f"{req.id:<5}{prompt_len:>7}{max_new_tokens:>8}"
            f"{len(req.generated_tokens):>11}{finish_step[req.id]:>15}"
        )

    print(f"\ntotal steps: {engine.now}")

    # Cheap correctness check: every request must have produced exactly the
    # number of tokens it asked for, and nothing may be left behind.
    assert not scheduler.waiting and not scheduler.running
    for req in scheduler.finished:
        assert len(req.generated_tokens) == req.max_new_tokens, req
        assert req.status == "finished", req
    print("checks passed: all requests finished with the right token count")


if __name__ == "__main__":
    main()
