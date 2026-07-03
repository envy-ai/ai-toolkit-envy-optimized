# Queue Drag Reorder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add drag handles to active queue tables so queued training jobs can be reordered while any running job remains pinned at the top.

**Architecture:** Reordering is persisted by a new queue API that validates queued jobs for one GPU and rewrites their `queue_position` values. The shared `JobsTable` renders active GPU sections through a sortable table wrapper; running and stopping jobs are kept first and are not draggable.

**Tech Stack:** Next.js API routes, Prisma, React, `@dnd-kit/core`, `@dnd-kit/sortable`, `@dnd-kit/utilities`, Python source-level regression tests.

---

### Task 1: Regression Tests

**Files:**
- Create: `testing/test_queue_reorder_drag_handles.py`

- [ ] **Step 1: Write source-level tests**

```python
import json
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

class QueueReorderDragHandleTests(unittest.TestCase):
    def test_dnd_dependencies_are_declared(self):
        package_json = json.loads((REPO_ROOT / "ui/package.json").read_text())
        deps = package_json["dependencies"]
        self.assertIn("@dnd-kit/core", deps)
        self.assertIn("@dnd-kit/sortable", deps)
        self.assertIn("@dnd-kit/utilities", deps)

    def test_reorder_api_validates_queued_jobs_and_rewrites_positions(self):
        route_source = (REPO_ROOT / "ui/src/app/api/queue/[queueID]/reorder/route.ts").read_text()
        self.assertIn("export async function PATCH", route_source)
        self.assertIn("orderedJobIds", route_source)
        self.assertIn("status: 'queued'", route_source)
        self.assertIn("gpu_ids: queueID", route_source)
        self.assertIn("submittedJobs.length !== orderedJobIds.length", route_source)
        self.assertIn("queue_position: (index + 1) * 1000", route_source)

    def test_jobs_table_uses_dnd_handles_and_pins_running_jobs(self):
        table_source = (REPO_ROOT / "ui/src/components/JobsTable.tsx").read_text()
        self.assertIn("DndContext", table_source)
        self.assertIn("SortableContext", table_source)
        self.assertIn("useSortable", table_source)
        self.assertIn("GripVertical", table_source)
        self.assertIn("reorderQueueJobs", table_source)
        self.assertIn("runningRows", table_source)
        self.assertIn("queuedRows", table_source)
        self.assertIn("job.status === 'queued'", table_source)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run red test**

Run: `python -m unittest testing.test_queue_reorder_drag_handles`

Expected: fail because the route, dnd dependencies, and table wiring do not exist.

### Task 2: Queue Reorder API

**Files:**
- Create: `ui/src/app/api/queue/[queueID]/reorder/route.ts`

- [ ] **Step 1: Add `PATCH` route**

The route reads `{ orderedJobIds: string[] }`, validates every job is `queued` on the target `gpu_ids`, rejects missing/mismatched jobs, and updates positions inside a Prisma transaction.

- [ ] **Step 2: Run focused test**

Run: `python -m unittest testing.test_queue_reorder_drag_handles`

Expected: dependency/table tests still fail, route assertions pass.

### Task 3: Drag UI

**Files:**
- Modify: `ui/package.json`
- Modify: `ui/package-lock.json`
- Modify: `ui/src/components/JobsTable.tsx`
- Modify: `ui/src/utils/queue.ts`

- [ ] **Step 1: Install dnd-kit packages**

Run: `npm install @dnd-kit/core @dnd-kit/sortable @dnd-kit/utilities`

- [ ] **Step 2: Add queue utility**

Add `reorderQueueJobs(queueID, orderedJobIds)` to `ui/src/utils/queue.ts`, calling `PATCH /api/queue/${queueID}/reorder`.

- [ ] **Step 3: Add sortable table body**

Render active GPU sections with a dnd-enabled table. Running/stopping rows are rendered first and not draggable. Queued rows get a `GripVertical` handle and persist the new queued order on drag end.

- [ ] **Step 4: Run focused test**

Run: `python -m unittest testing.test_queue_reorder_drag_handles`

Expected: pass.

### Task 4: Verification

**Files:**
- Verify all changed UI files.

- [ ] **Step 1: Run focused tests**

Run: `python -m unittest testing.test_queue_reorder_drag_handles testing.test_jobs_table_dataset_thumbnail testing.test_jobs_table_dataset_lightbox`

Expected: pass.

- [ ] **Step 2: Build UI**

Run: `npm run build` from `ui/`

Expected: exit 0. The existing optional `macos-temperature-sensor` warning may still appear.
