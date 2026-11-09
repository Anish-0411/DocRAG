import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inspect
import retrieval
import reranker

for module in (retrieval, reranker):
    print("\n" + "=" * 75)
    print("MODULE:", module.__name__)

    names = (
        "build_retrievers",
        "dense_search",
        "sparse_search",
        "reciprocal_rank_fusion",
        "retrieve_candidates",
        "rerank_candidates",
        "rerank_and_select",
    )

    for name in names:
        fn = getattr(module, name, None)
        if fn is None:
            continue

        print(f"\n--- {name}{inspect.signature(fn)} ---")
        print(inspect.getsource(fn))

