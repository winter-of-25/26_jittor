import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))


def load(batch):
    path = os.path.join(ROOT, "batch_probe_%d.json" % batch)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if value.get("status") == "PASS" else None


def main():
    six = load(6)
    eight = load(8)
    four = load(4)
    three = load(3)
    selected = None
    reason = ""
    if six is not None:
        selected = 6
        reason = "batch6 is the mandatory starting policy"
        if (
            eight is not None
            and six["gpu_memory_mib"] < 20000
            and eight["gpu_memory_mib"] < 22500
            and eight["samples_per_second"] > six["samples_per_second"]
        ):
            selected = 8
            reason = "batch8 improved real samples/s after batch6 stayed below 20 GiB"
    elif four is not None:
        selected = 4
        reason = "batch6 failed; fixed memory fallback batch4 passed"
    elif three is not None:
        selected = 3
        reason = "batch6/batch4 failed; final fixed memory fallback batch3 passed"
    if selected is None:
        raise RuntimeError("B24 exhausted batch6 -> batch4 -> batch3")
    with open(os.path.join(ROOT, "selected_batch.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(selected) + "\n")
    with open(os.path.join(ROOT, "batch_selection.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_batch": selected,
                "reason": reason,
                "probes": {str(batch): load(batch) for batch in (3, 4, 6, 8)},
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(selected)


if __name__ == "__main__":
    main()

