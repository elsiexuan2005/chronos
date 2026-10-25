import csv
import uuid
from pathlib import Path


class DataCollector:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.snapshots = []

    def record(self, features):
        snapshot = {
            "session_id": self.session_id,
            **features
        }
        self.snapshots.append(snapshot)

    def save(self, finished_on_time):
        if not self.snapshots:
            return

        label = int(finished_on_time)

        for snapshot in self.snapshots:
            snapshot["finished_on_time"] = label

        data_dir = Path(__file__).resolve().parent.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "training.csv"
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        with open(csv_path, "a", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.snapshots[0].keys()
            )

            if not file_exists:
                writer.writeheader()

            writer.writerows(self.snapshots)

        self.snapshots.clear()

        print(f"Training data saved to {csv_path}")


if __name__ == "__main__":
    collector = DataCollector()

    collector.record({
        "elapsed": 10,
        "progress": 0.2,
        "remaining": 80
    })

    collector.record({
        "elapsed": 20,
        "progress": 0.4,
        "remaining": 70
    })

    collector.save(finished_on_time=True)