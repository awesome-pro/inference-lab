"""Run the scheduler simulation on a set of test requests.

    ./.venv/bin/python -m experiments.scheduler_demo

Runs the engine to completion, then draws the whole run as a timeline: time
runs left to right, one column per step, and each row is one queue slot.
Reading a single column tells you the exact state of every queue at that
instant.

Not learning-critical: this is just a driver around Engine/Scheduler/Runner.
"""

from engine import Engine, Request, Runner, Scheduler

REQUEST_SPECS = [
    ("r0", 8, 3, 0.1),
    ("r1", 3, 5, 0.2),
    ("r2", 12, 2, 0.5),
    ("r3", 5, 4, 3.5),
    ("r4", 1, 1, 4.5),
    ("r5", 7, 3, 6.0),
]

MAX_RUNNING = 3
MAX_BATCH_TOKENS = 12   # per-step token budget: prefill costs prompt len, decode costs 1
STATIC = False   # True = hold each batch until it fully drains, no refilling
TRACE = False    # True = also print the verbose per-step queue dump

LABEL_WIDTH = 13


def make_requests():
    return [
        Request(
            id=req_id,
            prompt_tokens=list(range(prompt_len)),
            arrival_time=arrival_time,
            max_new_tokens=max_new_tokens,
        )
        for req_id, prompt_len, max_new_tokens, arrival_time in REQUEST_SPECS
    ]


def show_queues(scheduler):
    print("  waiting :", [r.id for r in scheduler.waiting] or "-")
    print("  running :", [r.id for r in scheduler.running] or "-")
    print("  finished:", [r.id for r in scheduler.finished] or "-")


def render_timeline(history, arrivals, max_running, max_batch_tokens):
    """Draw one column per step, one row per queue slot."""
    if not history:
        return

    # `waiting` holds every submitted request, including ones that have not
    # arrived yet. Split those out so a column shows only what is genuinely
    # queued right now.
    future = [[rid for rid in step["waiting"] if arrivals[rid] > step["time"]]
               for step in history]
    waiting = [[rid for rid in step["waiting"] if arrivals[rid] <= step["time"]]
               for step in history]
    running = [step["running"] for step in history]
    finished = [step["finished"] for step in history]
    times = [step["time"] for step in history]

    ids = [rid for step in history for rid in step["waiting"] + step["running"]]
    cell = max(len(rid) for rid in ids) + 2

    def cells(values):
        return "".join(f"{v:<{cell}}" for v in values)

    def band(label, queues, depth):
        for i in range(depth):
            values = [q[i] if i < len(q) else "" for q in queues]
            name = label if depth == 1 else f"{label}[{i}]"
            print(f"{name:<{LABEL_WIDTH}}{cells(values)}")

    print("=== timeline ===")
    print("one column per step, left to right")
    print("  future  submitted but arrival_time is still in the future")
    print("  waiting  arrived, queued, no free slot yet")
    print("  running  holding a slot; a blank row is a free slot")
    print("  finished completed requests, oldest first; they stay here once done")
    print(f"  tokens   batch tokens used per step (budget {max_batch_tokens})")
    print()
    print(f"{'t':<{LABEL_WIDTH}}{cells(str(t) for t in times)}")

    band("future", future, max((len(q) for q in future), default=0))

    print("-" * LABEL_WIDTH + "-" * (cell * len(history)))

    band("waiting", waiting, max((len(q) for q in waiting), default=0))

    print("-" * LABEL_WIDTH + "-" * (cell * len(history)))

    print(f"{'tokens':<{LABEL_WIDTH}}"
          f"{cells(str(step['tokens']) for step in history)}")
    band("running", running, max_running)

    print("-" * LABEL_WIDTH + "-" * (cell * len(history)))

    band("finished", finished, max((len(q) for q in finished), default=0))

    print()


def main():
    scheduler = Scheduler(
        max_running_requests=MAX_RUNNING,
        max_batch_tokens=MAX_BATCH_TOKENS,
        static=STATIC,
    )
    engine = Engine(scheduler, Runner())

    requests = make_requests()
    for request in requests:
        engine.submit(request)

    print(f"policy = {'static' if STATIC else 'continuous'}")
    print(f"max_running_requests = {MAX_RUNNING}, "
          f"max_batch_tokens = {MAX_BATCH_TOKENS}")
    for req_id, prompt_len, max_new_tokens, arrival_time in REQUEST_SPECS:
        print(f"  {req_id}  prompt={prompt_len:<3} max_new_tokens={max_new_tokens:<3} "
              f"arrives at t={arrival_time}")
    print()

    admit_time = {}         # when the request left the waiting queue
    first_token_time = {}   # when its first generated token came out
    finish_time = {}        # when its last token came out

    while scheduler.has_work():
        engine.step()

        for req in requests:
            # Status stops being "waiting" on the step it was admitted, and
            # step() has already advanced the clock, so that step was now-1.
            if req.status != "waiting" and req.id not in admit_time:
                admit_time[req.id] = engine.now - 1
            if req.generated_tokens and req.id not in first_token_time:
                first_token_time[req.id] = engine.now

        for req in scheduler.finished:
            finish_time.setdefault(req.id, engine.now)

        if TRACE:
            print(f"--- after step at t={engine.now - 1} ---")
            show_queues(scheduler)
            print()

    # One extra column for the state after the last step, otherwise the final
    # completion never appears in the finished band.
    engine.history.append(engine.snapshot())

    arrivals = {req_id: arrival for req_id, _, _, arrival in REQUEST_SPECS}
    render_timeline(engine.history, arrivals, MAX_RUNNING, MAX_BATCH_TOKENS)

    print("=== timing per request ===")
    print("  queue  arrival -> admitted (sat in the waiting queue)")
    print("  ttft   arrival -> first generated token (= queue + prefill + 1 decode)")
    print("  total  arrival -> last generated token")
    print()
    print(f"{'id':<5}{'arrive':>8}{'admit':>7}{'queue':>7}{'ttft':>7}"
          f"{'finish':>8}{'total':>7}")
    for req_id, _, _, arrival_time in REQUEST_SPECS:
        admit = admit_time[req_id]
        first = first_token_time[req_id]
        finish = finish_time[req_id]
        print(f"{req_id:<5}{arrival_time:>8}{admit:>7}{admit - arrival_time:>7}"
              f"{first - arrival_time:>7}{finish:>8}{finish - arrival_time:>7}")

    count = len(requests)
    print(f"\n{'mean':<5}{'':>8}{'':>7}"
          f"{sum(admit_time[r.id] - r.arrival_time for r in requests) / count:>7.2f}"
          f"{sum(first_token_time[r.id] - r.arrival_time for r in requests) / count:>7.2f}"
          f"{'':>8}"
          f"{sum(finish_time[r.id] - r.arrival_time for r in requests) / count:>7.2f}")

    print(f"\ntotal steps: {engine.now}")

    # Cheap correctness check: every request must have produced exactly the
    # number of tokens it asked for, and nothing may be left behind.
    assert not scheduler.waiting and not scheduler.running
    for req in scheduler.finished:
        assert len(req.generated_tokens) == req.max_new_tokens, req
        assert req.status == "finished", req

    # ...and no step may have overrun the token budget or the slot count.
    for step in engine.history:
        assert step["tokens"] <= MAX_BATCH_TOKENS, step
        assert len(step["running"]) <= MAX_RUNNING, step
    print("checks passed: all requests finished with the right token count, "
          "and no step exceeded the slot or token budget")


if __name__ == "__main__":
    main()
