from citation_service import finalize_citations

evidence = [
    {
        "evidence_id": "EVD-AAAAAAAAAAAA",
        "chunk_id": "chunk-a",
        "document_id": "doc-a",
        "document_version": "v1",
        "page": 1,
        "file_name": "report_a.pdf",
        "content": "The aircraft encountered calm winds.",
    },
    {
        "evidence_id": "EVD-BBBBBBBBBBBB",
        "chunk_id": "chunk-b",
        "document_id": "doc-b",
        "document_version": "v1",
        "page": 2,
        "file_name": "report_b.pdf",
        "content": "Visibility was 10 miles.",
    },
]

tests = [
    (
        "Valid citation",
        "Calm winds were reported [EVD-AAAAAAAAAAAA].",
    ),
    (
        "Invalid citation",
        "Calm winds were reported [EVD-CCCCCCCCCCCC].",
    ),
    (
        "Missing citation",
        "The weather was clear and calm.",
    ),
    (
        "Mixed valid and invalid citations",
        "Calm winds [EVD-AAAAAAAAAAAA] and rain "
        "[EVD-CCCCCCCCCCCC].",
    ),
    (
        "Full-width citation brackets",
        "Visibility was 10 miles 【EVD-BBBBBBBBBBBB】.",
    ),
    (
        "Exact abstention",
        "I don't have enough information in the provided documents.",
    ),
]

for name, answer in tests:
    result = finalize_citations(answer, evidence)

    print("\n" + "=" * 65)
    print(name)
    print("Input:              ", answer)
    print("Final answer:       ", result["answer"])
    print("Retrieved:          ", result["retrieved_evidence_count"])
    print("Cited:              ", result["cited_evidence_count"])
    print("Uncited:            ", result["uncited_evidence_count"])
    print("Citation flags:     ", [
        (c["evidence_id"], c["cited"])
        for c in result["citations"]
    ])

print("\nPhase 6D deterministic validation completed.")
print("No production files modified.")
