import json
import os

import dotenv
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from pymongo import MongoClient

dotenv.load_dotenv()

REPO_ID = "ZefanCai/Open-Jev"
CONFIG = "release-v2-redistributable"
SPLITS = ["train", "validation", "test", "ood", "calibration"]
DB_NAME = "open_jev"
JSON_SUFFIX = "_json"
# record_json duplicates every other column in the row, so it is dropped
# rather than expanded into the document.
DROPPED_COLUMNS = {"record_json"}


def mongo_client() -> MongoClient:
    uri = (
        f"mongodb://{os.environ['MONGODB_USERNAME']}:{os.environ['MONGODB_PASSWORD']}"
        f"@{os.environ['MONGODB_HOST']}:{os.environ['MONGODB_PORT']}/?authSource=admin"
    )
    return MongoClient(uri)


def load_split(split: str) -> list[dict]:
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"data/{CONFIG}/{split}-00000-of-00001.parquet",
    )
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["_id"] = row["id"]
        for key in list(row):
            if key in DROPPED_COLUMNS:
                del row[key]
            elif key.endswith(JSON_SUFFIX) and isinstance(row[key], str):
                new_key = key[: -len(JSON_SUFFIX)]
                row[new_key] = json.loads(row[key])
                del row[key]
    return rows


def main() -> None:
    client = mongo_client()
    db = client[DB_NAME]

    for split in SPLITS:
        print(f"Downloading split '{split}'...")
        rows = load_split(split)

        collection = db[split]
        collection.drop()
        if rows:
            collection.insert_many(rows, ordered=False)
        print(f"Loaded {len(rows)} documents into {DB_NAME}.{split}")

    client.close()


if __name__ == "__main__":
    main()
