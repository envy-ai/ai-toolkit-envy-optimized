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
