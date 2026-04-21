import argparse
import os
import sqlite3


def stop_training_jobs(db_path="./aitk_db.db", info="Job stopped"):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found: {db_path}")

    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute(
                """
                UPDATE Job
                SET
                    status = 'stopped',
                    stop = 1,
                    return_to_queue = 0,
                    info = ?,
                    pid = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE job_type = 'train'
                """,
                (info,),
            )
            changed = cursor.rowcount
            cursor.execute("COMMIT")
            return changed
        except Exception:
            cursor.execute("ROLLBACK")
            raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Mark all training jobs in the AI Toolkit SQLite database as stopped."
    )
    parser.add_argument(
        "--db",
        default="./aitk_db.db",
        help="Path to the SQLite database. Defaults to ./aitk_db.db",
    )
    parser.add_argument(
        "--info",
        default="Job stopped",
        help="Info message to write to each stopped training job.",
    )
    args = parser.parse_args()

    changed = stop_training_jobs(args.db, args.info)
    print(f"Marked {changed} training job(s) as stopped in {args.db}")


if __name__ == "__main__":
    main()
