"""Builds the offline corpus for the worked example.

Deterministic: rerunning it reproduces traces.jsonl byte for byte. The corpus is
entirely invented — it derives from no real newsletter and contains no real data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

TOPICS = ["model-releases", "agents", "infrastructure", "safety", "research"]
SUBJECTS = [
    ("Northwind Labs", "released a 12B open-weights model"),
    ("Calder AI", "cut inference cost by a third"),
    ("the Solmark AI Office", "opened consultation on agent disclosure"),
    ("Pellworm Robotics", "showed a warehouse manipulation policy"),
    ("Basalt Research", "published a paper on retrieval collapse"),
]
FAILURE_MODES = ["clean", "merged_stories", "hallucinated_source", "wrong_topic", "promo_not_filtered"]


def build(n: int = 400, seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        mode = FAILURE_MODES[i % len(FAILURE_MODES)]
        org, event = rng.choice(SUBJECTS)
        other_org, other_event = rng.choice(SUBJECTS)
        topic = rng.choice(TOPICS)

        newsletter = (
            f"THE WIRE — issue {i}\n\n"
            f"{org} {event}. The announcement was made on the company blog.\n\n"
            f"{other_org} {other_event}. Details are in their engineering notes.\n\n"
            f"SPONSORED: Try Kettlewell Cloud free for 30 days."
        )

        if mode == "merged_stories":
            stories = [
                {"headline": f"{org} {event} and {other_org} {other_event}",
                 "topic": topic, "sources": [f"{org} blog"]}
            ]
        elif mode == "hallucinated_source":
            stories = [
                {"headline": f"{org} {event}", "topic": topic,
                 "sources": [f"{org} blog", "Meridian Wire exclusive"]},
                {"headline": f"{other_org} {other_event}", "topic": topic,
                 "sources": [f"{other_org} notes"]},
            ]
        elif mode == "wrong_topic":
            wrong = next(t for t in TOPICS if t != topic)
            stories = [
                {"headline": f"{org} {event}", "topic": wrong, "sources": [f"{org} blog"]},
                {"headline": f"{other_org} {other_event}", "topic": topic,
                 "sources": [f"{other_org} notes"]},
            ]
        elif mode == "promo_not_filtered":
            stories = [
                {"headline": f"{org} {event}", "topic": topic, "sources": [f"{org} blog"]},
                {"headline": "Try Kettlewell Cloud free for 30 days",
                 "topic": topic, "sources": ["sponsor"]},
            ]
        else:
            stories = [
                {"headline": f"{org} {event}", "topic": topic, "sources": [f"{org} blog"]},
                {"headline": f"{other_org} {other_event}", "topic": topic,
                 "sources": [f"{other_org} notes"]},
            ]

        rows.append(
            {
                "trace_id": f"news-{i:03d}",
                "input": newsletter,
                "output": json.dumps({"stories": stories}, indent=2),
                "planted_mode": mode,
            }
        )
    return rows


if __name__ == "__main__":
    out = Path(__file__).parent / "traces.jsonl"
    out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in build()) + "\n")
    print(f"wrote {out}")
