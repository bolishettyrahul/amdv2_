# Phase 2 — Learned Hybrid Router (Not Implemented)

This package is intentionally empty. Phase 2 is a LightGBM classifier trained on Phase 1's logged JSONL dataset (features: domain, prompt embedding, task-shape signals; label: cheapest tier verified/critiqued as correct), deployed in hybrid mode where low-confidence predictions still fall back through the full Phase 1 cascade.

Phase 2 has a hard dependency on Phase 1 producing real logged task data first — it cannot be built or tested before that data exists. It is only pursued if time and API credits remain after Phase 1 is complete and running. See `../../plan/master-plan.md` for full detail.
