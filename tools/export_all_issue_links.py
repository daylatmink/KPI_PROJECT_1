#!/usr/bin/env python3
"""
Export all issue links from MongoDB to a single CSV file.
"""

import argparse
import csv
import os

from pymongo import MongoClient

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_DB = "JiraReposAnon"
DEFAULT_COLLECTION = "Apache"


def export_links(mongo_uri: str, db_name: str, collection: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    client = MongoClient(mongo_uri)
    db = client[db_name]
    col = db[collection]

    projection = {
        "key": 1,
        "fields.issuelinks": 1,
    }

    cursor = col.find({}, projection)

    with open(out_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=[
                "from_issue_key",
                "to_issue_key",
                "link_type",
                "raw_link_type",
                "direction",
            ],
        )
        writer.writeheader()

        total_links = 0
        for doc in cursor:
            self_key = doc.get("key")
            fields = doc.get("fields") or {}
            links = fields.get("issuelinks") or []
            for link in links:
                ltype = link.get("type") or {}
                raw_name = ltype.get("name")
                inward_desc = ltype.get("inward")
                outward_desc = ltype.get("outward")

                inward_issue = link.get("inwardIssue")
                if inward_issue:
                    other_key = inward_issue.get("key")
                    writer.writerow(
                        {
                            "from_issue_key": other_key,
                            "to_issue_key": self_key,
                            "link_type": inward_desc or raw_name,
                            "raw_link_type": raw_name,
                            "direction": "inward",
                        }
                    )
                    total_links += 1

                outward_issue = link.get("outwardIssue")
                if outward_issue:
                    other_key = outward_issue.get("key")
                    writer.writerow(
                        {
                            "from_issue_key": self_key,
                            "to_issue_key": other_key,
                            "link_type": outward_desc or raw_name,
                            "raw_link_type": raw_name,
                            "direction": "outward",
                        }
                    )
                    total_links += 1

    client.close()
    print(f"Saved: {out_path}")
    print(f"Total links exported: {total_links}")


def main():
    parser = argparse.ArgumentParser(description="Export all issue links to CSV.")
    parser.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--out",
        default="data/raw/all_issue_links.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    export_links(args.mongo_uri, args.db, args.collection, os.path.abspath(args.out))


if __name__ == "__main__":
    main()
