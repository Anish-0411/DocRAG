import csv
import json
from pathlib import Path
from collections import defaultdict

SNAPSHOT = Path(
    "index_store/snapshots/snapshot_20261101T075233_ddc94958"
)

GOLD_PATH = Path("evaluation/gold_questions.jsonl")

RESULTS_PATH = Path(
    "evaluation/results/phase3_candidates_20261103_171559/candidate_rankings.csv"
)

TARGET_IDS = ["Q02", "Q03", "Q06", "Q25", "Q36"]
EXPERIMENT = "expanded_100_100_100"


def load_gold_questions():
    questions = {}

    with GOLD_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)
                if item["question_id"] in TARGET_IDS:
                    questions[item["question_id"]] = item

    return questions


def load_chunks():
    chunks = {}

    path = SNAPSHOT / "chunks.jsonl"

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            metadata = item.get("metadata", {})
            chunk_id = metadata.get("chunk_id")

            if chunk_id:
                chunks[chunk_id] = item

    return chunks


def load_rankings():
    rankings = defaultdict(list)

    with RESULTS_PATH.open(
        newline="",
        encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            if (
                row["experiment"] == EXPERIMENT
                and row["question_id"] in TARGET_IDS
            ):
                key = (row["question_id"], row["stage"])
                rankings[key].append(row)

    for rows in rankings.values():
        rows.sort(key=lambda row: int(row["rank"]))

    return rankings


def print_chunk(title, chunk, extra=""):
    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)

    if extra:
        print(extra)

    if not chunk:
        print("FULL CHUNK TEXT NOT FOUND")
        return

    metadata = chunk.get("metadata", {})

    print("Chunk ID:", metadata.get("chunk_id"))
    print("Document:", metadata.get("file_name"))
    print("Page (zero-based):", metadata.get("page"))
    print("Paragraph:", metadata.get("paragraph_index"))
    print("Chunk index:", metadata.get("chunk_index"))

    print("\nFULL TEXT:\n")
    print(chunk.get("page_content", ""))


def main():
    questions = load_gold_questions()
    chunks = load_chunks()
    rankings = load_rankings()

    print("PHASE 3 EVIDENCE INSPECTION")
    print("Snapshot:", SNAPSHOT.name)
    print("Experiment:", EXPERIMENT)
    print("Questions:", len(questions))
    print("Chunks loaded:", len(chunks))

    for qid in TARGET_IDS:
        question = questions[qid]
        gold_ids = set(question["gold_chunk_ids"])

        print("\n\n" + "=" * 100)
        print("QUESTION:", qid)
        print("=" * 100)

        print("\nQUESTION TEXT:\n", question["question"])
        print("\nEXPECTED ANSWER:\n", question["expected_answer"])
        print("\nCATEGORY:", question.get("category"))
        print("EXPECTED DOCUMENT:", question.get("document"))
        print("EXPECTED PAGE (zero-based):", question.get("page"))

        # Full gold evidence
        print("\n\n######## GOLD EVIDENCE ########")

        for index, chunk_id in enumerate(gold_ids, start=1):
            print_chunk(
                f"GOLD CHUNK {index}",
                chunks.get(chunk_id),
                extra=f"Gold ID: {chunk_id}"
            )

        # Top five reranked candidates
        reranked = rankings.get((qid, "reranked"), [])

        print("\n\n######## TOP 5 RERANKED CANDIDATES ########")

        for row in reranked[:5]:
            chunk_id = row["chunk_id"]
            chunk = chunks.get(chunk_id)

            print_chunk(
                f"RERANKED RANK {row['rank']}",
                chunk,
                extra=(
                    f"Gold match: {row['is_gold_chunk']}\n"
                    f"Rerank score: {row['rerank_score']}\n"
                    f"RRF score: {row['rrf_score']}"
                )
            )

        # Gold positions across stages
        print("\n\n######## GOLD RANK POSITIONS ########")

        for stage in ["dense", "bm25", "rrf", "reranked", "context_top5"]:
            rows = rankings.get((qid, stage), [])

            positions = [
                row["rank"]
                for row in rows
                if row["chunk_id"] in gold_ids
            ]

            print(f"{stage:15s}: {positions or 'NOT FOUND'}")


if __name__ == "__main__":
    main()
