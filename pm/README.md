# Project management

Single source of truth for the wrap-up and refinement of the calibration-aware TTA project.
Everything an agent or a person needs to pick up work lives here, tracked in git.

| File | What it holds | Who updates it |
|---|---|---|
| `PLAN.md` | Goal, phases, deliverables, acceptance criteria | lead, on approval only |
| `RULES.md` | Standing rules every task must follow | lead, on approval only |
| `KNOWLEDGE.md` | Established facts with evidence pointers (the knowledge base) | anyone, with a source |
| `DECISIONS.md` | Dated decision log: what, why, who | lead |
| `STATUS.md` | Current state, next steps, blockers, reminders | lead, every session |
| `tasks/Txx-*.md` | One file per task: goal, inputs, outputs, acceptance, status | task owner |
| `handoffs/` | One note per completed task or session: what changed, what is open | whoever finishes |
| `research/` | Literature notes, formalizations, figure mappings | anyone |

Workflow:
1. Pick a task whose dependencies are `done` in `tasks/`. Set its status to `in progress`.
2. Follow `RULES.md`. Record any new fact in `KNOWLEDGE.md` with its source.
3. Finish with a handoff note in `handoffs/` and set the task to `review`.
4. A reviewer who did not do the work checks it against the task's acceptance criteria;
   only then is it `done`.
5. The lead updates `STATUS.md` and, for anything that changes scope, `DECISIONS.md`.

Earlier planning documents (pre-PM): `docs/plans/2026-11-04-reservoir-cache-label-free-temperature.md`,
`docs/plans/TASKS.md` (backlog for *after* the wrap-up), `docs/baseline_hyperparameter_audit.md`.
