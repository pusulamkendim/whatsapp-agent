import argparse
import json
import sys

from app.database import SessionLocal, init_db
from app.models import Agent
from app.rag.indexing import reindex_all
from app.rag.retrieval import retrieve_agent_context


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG admin utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reindex_parser = subparsers.add_parser("reindex-all")
    reindex_parser.add_argument("--embedding-model", default=None)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--file", default="rag_eval_set.json")
    eval_parser.add_argument("--reindex", action="store_true")

    args = parser.parse_args()
    init_db()
    db = SessionLocal()
    try:
        if args.command == "reindex-all":
            result = reindex_all(db, model_ref=args.embedding_model)
            db.commit()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("ok") else 1
        if args.command == "eval":
            if args.reindex:
                reindex_all(db)
                db.commit()
            result = run_eval(args.file, db)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["accuracy"] >= 0.8 else 1
    finally:
        db.close()
    return 1


def run_eval(path: str, db) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = []
    passed = 0
    for case in cases:
        agent = db.query(Agent).filter(Agent.slug == case["agent_slug"], Agent.active == True).first()
        if not agent:
            results.append({**case, "ok": False, "error": "agent_not_found"})
            continue
        result = retrieve_agent_context(
            agent,
            case["query"],
            db,
            external_user_id="rag-eval",
            top_k=case.get("top_k"),
            final_chunks=case.get("final_chunks"),
        )
        context = result.context.lower()
        expected_terms = [term.lower() for term in case.get("expected_terms", [])]
        forbidden_terms = [term.lower() for term in case.get("forbidden_terms", [])]
        ok = all(term in context for term in expected_terms) and not any(term in context for term in forbidden_terms)
        passed += 1 if ok else 0
        results.append({
            "name": case.get("name", case["query"]),
            "agent_slug": case["agent_slug"],
            "query": case["query"],
            "ok": ok,
            "chunk_ids": [chunk.chunk_id for chunk in result.chunks],
            "scores": [round(chunk.score, 4) for chunk in result.chunks],
        })
    db.commit()
    total = len(cases)
    return {
        "ok": passed == total,
        "passed": passed,
        "total": total,
        "accuracy": round(passed / total, 4) if total else 0,
        "results": results,
    }


if __name__ == "__main__":
    sys.exit(main())
