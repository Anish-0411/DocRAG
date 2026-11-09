
from pathlib import Path
from datetime import datetime

REPORT_PATH = Path(__file__).resolve().parent / "PHASE3_FINAL_REPORT.md"

report = r"""
# CortexRAG — Phase 3: Retrieval Optimization and Validation

## 1. Executive Summary

Phase 3 focused on improving CortexRAG's retrieval pipeline through
candidate-depth experiments, cross-encoder reranking, neighbor-context
expansion, and query expansion.

The experiments used the existing indexed knowledge base containing
12,124 chunks and a manually curated 40-question gold evaluation dataset.

All Phase 3 experiments were performed offline:
- No Groq calls.
- No external LLM or generation API calls.
- No production retrieval configuration changes.
- No modification of the active knowledge-base snapshot.

**Conclusion:** Candidate-depth optimization improved retrieval metrics.
However, blanket neighbor expansion introduced substantial irrelevant
context and is not approved for production integration.

---

## 2. Starting Baseline — Phase 1

The original production retrieval configuration was evaluated against
40 questions.

| Metric | Baseline |
|---|---:|
| Recall@5 | 0.7000 |
| Recall@10 | 0.8125 |
| Recall@20 | 0.8125 |
| Precision@5 | 0.1450 |
| MRR@20 | 0.6217 |
| nDCG@10 | 0.6659 |

The baseline used:
- Dense retrieval.
- BM25 sparse retrieval.
- Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking.

The dense retriever outperformed BM25 independently, while hybrid
retrieval improved early evidence coverage.

---

## 3. Phase 3A — Candidate-Depth Experiments

Two candidate-depth configurations were tested:

| Configuration | Dense | BM25 | RRF |
|---|---:|---:|---:|
| Expanded 100 | 100 | 100 | 100 |
| Expanded 200 | 200 | 200 | 200 |

Both configurations used the existing cross-encoder reranker.

### 3.1 Results

| Metric | Depth 100 | Depth 200 |
|---|---:|---:|
| Recall@5 | 0.7500 | 0.7500 |
| Recall@10 | 0.8875 | 0.8875 |
| Recall@20 | 0.9500 | 0.9500 |
| Precision@5 | 0.1550 | 0.1550 |
| Precision@10 | 0.0925 | 0.0925 |
| Precision@20 | 0.0500 | 0.0500 |
| MRR@20 | 0.6759 | 0.6761 |
| nDCG@10 | 0.7221 | 0.7221 |
| Context top-5 gold coverage | 75% | 75% |

### 3.2 Comparison with the original baseline

| Metric | Phase 1 | Phase 3 depth 100 |
|---|---:|---:|
| Recall@5 | 0.7000 | 0.7500 |
| Recall@10 | 0.8125 | 0.8875 |
| Recall@20 | 0.8125 | 0.9500 |
| Precision@5 | 0.1450 | 0.1550 |
| MRR@20 | 0.6217 | 0.6759 |
| nDCG@10 | 0.6659 | 0.7221 |

The depth-100 experiment improved all reported comparison metrics.

Depth 200 offered practically no meaningful improvement over depth 100.
Therefore, depth 100 is the preferred experimental configuration.

**Important:** These are offline retrieval experiment results, not
production-deployment results. The configurations have not yet been
integrated into the application.

---

## 4. Phase 3B — Neighbor Context Expansion

The experiment examined whether adjacent chunks from the same document
could recover useful evidence missing from the original top-five context.

### 4.1 Experimental method

- Started with the original top-five context chunks.
- Identified previous and next chunks from the same document and version.
- Avoided duplicate chunk IDs.
- Preserved original chunks before adding neighbors.
- Applied a 5,000-token budget using the `cl100k_base` tokenizer.
- Evaluated exact gold-chunk coverage across 40 questions.

### 4.2 Results

| Metric | Original | Expanded |
|---|---:|---:|
| Exact gold context coverage | 30/40 (75%) | 32/40 (80%) |
| Average context chunks | 5 | 13.3 |
| Average context tokens | — | 1,739.88 |

Two additional questions recovered exact gold evidence:
- Q24
- Q36

No previously covered question lost its exact gold evidence.

### 4.3 Important limitation

The neighbor experiment used a 5,000-token budget calculated with
`tiktoken` and allowed more than five chunks.

Production's existing context selection uses a different approximate
token-counting method and a top-five selection limit.

Therefore, the 80% result must not be interpreted as a production
configuration improvement. It is an experimental context-coverage result.

---

## 5. Phase 3B — Qualitative Evidence Audit

A detailed audit was conducted for ten difficult questions:

Q02, Q03, Q06, Q07, Q08, Q23, Q24, Q25, Q27 and Q36.

The audit included:
- 50 original top-five chunks.
- 90 additional neighbor chunks.
- 140 total context entries.

### 5.1 Findings

| Question | Finding | Assessment |
|---|---|---|
| Q02 | Mostly unrelated accident-report context | Noisy |
| Q03 | Unrelated reports and boilerplate | Noisy |
| Q06 | Some useful same-document context, alongside references | Mixed |
| Q07 | Mostly unrelated DHS sections | Noisy |
| Q08 | Repeated LangChain content | Redundant |
| Q23 | Adjacent chunk contains the probable-cause information | Useful |
| Q24 | Neighbor contains departure and destination fields | Useful |
| Q25 | Mostly unrelated flight records | Noisy |
| Q27 | Mostly investigation-docket boilerplate | Noisy |
| Q36 | Neighbors contain greenhouse-gas reduction measures | Useful |

### 5.2 Decision

Blanket neighbor expansion across all five original context chunks
is not recommended.

Although it improved exact gold-context coverage from 75% to 80%,
it introduced many irrelevant or repetitive chunks.

The experiment demonstrates that adjacent chunks can be valuable,
but expansion should be selective and validated before production use.

---

## 6. Phase 3C — Query Expansion

A deterministic terminology-expansion experiment was conducted.

Only Q25 received query expansion, using aviation route terminology.

The original and expanded configurations produced no measurable
difference in the reported retrieval metrics or top-five context
coverage.

The experiment did not justify adding query expansion to the
current retrieval pipeline.

**Decision: Discard the tested query-expansion approach.**

Note: The initial query-expansion script's MRR calculation scanned
the full reranked candidate list rather than restricting evaluation
to the first 20 ranks. Its reported MRR should therefore not be
presented as MRR@20. The Phase 3A MRR@20 result is used instead.

---

## 7. Final Phase 3 Decisions

| Component | Decision |
|---|---|
| Dense retrieval | Retain |
| BM25 retrieval | Retain |
| Reciprocal Rank Fusion | Retain |
| Cross-encoder reranking | Retain |
| Candidate depth 100 | Preferred experimental configuration |
| Candidate depth 200 | Discard; negligible additional benefit |
| Blanket neighbor expansion | Do not integrate |
| Selective neighbor expansion | Future optimization candidate |
| Tested query expansion | Discard |

No production configuration has been changed as a result of these
experiments.

---

## 8. Limitations and Outstanding Validation

1. The gold dataset contains 40 questions and should be expanded for
   stronger evaluation confidence.
2. Exact chunk-ID matching can undercount semantically useful adjacent
   evidence.
3. Candidate-depth experiments have not been validated through
   end-to-end answer generation.
4. Neighbor expansion has not been tested against answer correctness,
   faithfulness, citation support, and latency using the actual
   production context-selection implementation.
5. No production memory or latency profiling has been completed for
   the proposed depth-100 configuration.
6. The final offline local-model architecture remains a separate
   engineering requirement.

Retrieval metrics alone do not establish answer quality.

---

## 9. Artifacts Produced

### Phase 1
- `gold_questions.jsonl`
- `validate_gold_dataset.py`
- `run_retrieval_benchmark.py`
- `results/baseline_20261103_154939/`

### Phase 2
- `phase2_answers.jsonl`
- `capture_phase2_answers.py`
- `evaluate_phase2.py`
- `phase2_question_report.csv`
- `phase2_report.md`

### Phase 3
- `run_phase3_candidate_experiment.py`
- `run_phase3_neighbor_experiment.py`
- `run_phase3_query_expansion.py`
- `audit_phase3_neighbor_context.py`
- `results/phase3_candidates_20261103_171559/`
- `results/phase3_neighbor_experiment/`
- `results/phase3_query_expansion/`
- `results/phase3_neighbor_audit/neighbor_context_audit.csv`

---

## 10. Phase 3 Completion Status

| Activity | Status |
|---|---|
| Candidate-depth experiments | Complete |
| Depth-100 vs depth-200 comparison | Complete |
| Neighbor expansion experiment | Complete |
| Neighbor evidence audit | Complete |
| Query expansion experiment | Complete |
| Retrieval strategy decisions | Complete |
| Production integration | Pending |
| End-to-end answer-quality validation of new configuration | Pending |
| Latency and memory validation | Pending |
| Regression testing after integration | Pending |

**Phase 3 retrieval experimentation and evidence auditing are complete.**
Production integration and final end-to-end validation remain
deliberately outside this phase's completed experimental work.

---

## 11. Next Engineering Stage

The next stage should focus on improving the complete CortexRAG
application rather than continuing uncontrolled retrieval tuning.

Recommended sequence:

1. Review the existing Phase 2 answer-quality results.
2. Finalize the retrieval configuration to be tested.
3. Implement and test selective evidence handling only if justified.
4. Improve citation precision and citation-to-source traceability.
5. Evaluate answer faithfulness, correctness, relevance and abstention.
6. Measure latency and memory consumption.
7. Run the complete 40-question regression suite.
8. Continue toward a fully offline, local-model-based architecture.

The temporary Groq development integration must not become a
dependency of the final offline application.

---

## Final Conclusion

Phase 3 established that increasing candidate retrieval depth to
100 and retaining cross-encoder reranking improves retrieval metrics
on the current 40-question evaluation set.

Neighbor expansion can recover useful adjacent evidence, but
unrestricted expansion creates substantial noise. Query expansion
did not produce measurable gains.

The evidence supports retaining the existing hybrid retrieval
architecture and carrying candidate depth 100 forward as an
experimental configuration for subsequent validation.

**Phase 3 retrieval experiments and qualitative evidence audit:
COMPLETE.**
"""


def main():
    REPORT_PATH.write_text(report.strip() + "\n", encoding="utf-8")

    print("PHASE 3 FINAL REPORT GENERATED")
    print(f"Path: {REPORT_PATH}")
    print(f"Size: {REPORT_PATH.stat().st_size:,} bytes")
    print(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    print("\nReport sections:")

    for line in report.splitlines():
        if line.startswith("# "):
            print(line)


if __name__ == "__main__":
    main()
