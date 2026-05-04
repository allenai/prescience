"""Leakage-safe tool implementations for the GPT-5 agent baseline.

Provides a `ToolContext` (per-target view of the corpus + cutoff filter) and four
tool functions that an LLM can invoke through OpenAI function-calling. Every tool
result passes through `_is_leaking` which drops the target paper itself, any
post-cutoff paper, any paper whose title matches the target's, and any paper
whose abstract overlaps the target's abstract by 50 chars. The agent never
receives the target's content via any tool.

GRIT (GritLM-7B) is used for query-side embedding for `search_papers`. A FAISS
index over the corpus's GRIT embeddings is shared across workers; query embedding
is serialized through a single GPU via `gritlm_lock`.
"""

import json
import threading
from typing import Dict, List, Optional

import numpy as np


# OpenAI tool/function schemas, in `tools` parameter format.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Semantic search over papers in the corpus published strictly before the target's "
                "cutoff date. Returns up to `limit` results (default 10, max 50), each with "
                "corpus_id, title, abstract, date, and authors. Sorted by relevance to the query. "
                "Will not surface the target paper or any post-cutoff paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text query (a topic, method, or claim)."},
                    "limit": {"type": "integer", "description": "Max results to return.", "default": 10}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper",
            "description": (
                "Look up a single pre-cutoff paper by corpus_id. Returns its title, abstract, date, "
                "authors, and the corpus_ids of its key (influential) references. Returns null if the "
                "paper is post-cutoff, hidden, or unknown."
            ),
            "parameters": {
                "type": "object",
                "properties": {"corpus_id": {"type": "integer"}},
                "required": ["corpus_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_author_recent_papers",
            "description": (
                "Get an author's most recent publications before the cutoff date (most recent first), "
                "each with corpus_id, title, abstract, and date. Useful for understanding what a target "
                "paper's authors have been working on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "author_id": {"type": "string", "description": "Disambiguated author_id (S2AND format)."},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["author_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_references_of",
            "description": (
                "Get the influential (key) references of a pre-cutoff paper, each with corpus_id, "
                "title, abstract, and date. References that are themselves post-cutoff or hidden are "
                "filtered out. Returns null if the paper is unknown or hidden."
            ),
            "parameters": {
                "type": "object",
                "properties": {"corpus_id": {"type": "integer"}},
                "required": ["corpus_id"]
            }
        }
    },
]


class ToolContext:
    """Per-target context: cutoff, target metadata, plus shared corpus / FAISS / GRIT.

    Multiple ToolContexts (one per concurrent target) share the same `corpus`,
    `faiss_index`, `faiss_id_map`, and `gritlm_runtime`. Only `cutoff_date`,
    `target_corpus_id`, and target text fields differ per-instance.
    """

    def __init__(
        self,
        corpus: "CorpusView",
        gritlm_runtime: "GritLMRuntime",
        cutoff_date: str,
        target_corpus_id: int,
        target_title: str,
        target_abstract: str,
    ):
        self.corpus = corpus
        self.gritlm = gritlm_runtime
        self.cutoff_date = cutoff_date
        self.target_corpus_id = target_corpus_id
        self.target_title_lc = (target_title or "").strip().lower()
        self.target_abstract_lc = (target_abstract or "").strip().lower()
        # Pre-build allowed corpus_ids (set lookup is fast)
        self.candidate_ids: frozenset = frozenset(
            cid for cid, p in corpus.papers_dict.items()
            if p.get("date", "") < cutoff_date and cid != target_corpus_id
        )

    # -- leakage filter --------------------------------------------------
    def _is_leaking(self, paper: Dict) -> bool:
        if paper is None:
            return True
        cid = paper.get("corpus_id")
        if cid == self.target_corpus_id:
            return True
        if cid not in self.candidate_ids:
            return True
        title_lc = (paper.get("title") or "").strip().lower()
        if title_lc and self.target_title_lc:
            if title_lc == self.target_title_lc or title_lc in self.target_title_lc or self.target_title_lc in title_lc:
                return True
        abstract_lc = (paper.get("abstract") or "").lower()
        if abstract_lc and self.target_abstract_lc and len(self.target_abstract_lc) >= 50:
            head = self.target_abstract_lc[:50]
            tail = self.target_abstract_lc[-50:]
            if head in abstract_lc or tail in abstract_lc:
                return True
            if len(abstract_lc) >= 50 and (abstract_lc[:50] in self.target_abstract_lc or abstract_lc[-50:] in self.target_abstract_lc):
                return True
        return False

    # -- output shaping --------------------------------------------------
    def _shape_paper(self, paper: Dict, include_refs: bool = False) -> Optional[Dict]:
        if self._is_leaking(paper):
            return None
        out = {
            "corpus_id": paper["corpus_id"],
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "date": paper.get("date"),
        }
        authors = paper.get("authors") or []
        out["authors"] = [
            {"author_id": a.get("author_id"), "name": a.get("name")}
            for a in authors if isinstance(a, dict) and a.get("author_id")
        ]
        if include_refs:
            refs = paper.get("key_references") or []
            out["key_references"] = [r["corpus_id"] for r in refs if isinstance(r, dict) and "corpus_id" in r]
        return out


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_papers(query: str, ctx: ToolContext, limit: int = 10) -> List[Dict]:
    if not isinstance(query, str) or not query.strip():
        return []
    limit = max(1, min(int(limit), 50))
    over = limit * 5
    q_emb = ctx.gritlm.embed_query(query)  # (D,) float32
    distances, indices = ctx.corpus.faiss_index.search(q_emb.reshape(1, -1).astype(np.float32), over)
    out = []
    for idx in indices[0]:
        if idx < 0:
            continue
        cid = ctx.corpus.faiss_id_map[idx]
        paper = ctx.corpus.papers_dict.get(cid)
        if paper is None:
            continue
        shaped = ctx._shape_paper(paper)
        if shaped is None:
            continue
        out.append(shaped)
        if len(out) >= limit:
            break
    return out


def get_paper(corpus_id: int, ctx: ToolContext) -> Optional[Dict]:
    paper = ctx.corpus.papers_dict.get(int(corpus_id))
    if paper is None:
        return None
    return ctx._shape_paper(paper, include_refs=True)


def get_author_recent_papers(author_id: str, ctx: ToolContext, limit: int = 10) -> List[Dict]:
    if not isinstance(author_id, str):
        return []
    limit = max(1, min(int(limit), 50))
    pubs = ctx.corpus.sd2publications.get(author_id) or []
    pubs = [c for c in pubs if c in ctx.candidate_ids]
    pubs.sort(key=lambda c: ctx.corpus.papers_dict[c].get("date", ""), reverse=True)
    out = []
    for cid in pubs:
        paper = ctx.corpus.papers_dict.get(cid)
        if paper is None:
            continue
        shaped = ctx._shape_paper(paper)
        if shaped is None:
            continue
        out.append(shaped)
        if len(out) >= limit:
            break
    return out


def get_references_of(corpus_id: int, ctx: ToolContext) -> Optional[List[Dict]]:
    paper = ctx.corpus.papers_dict.get(int(corpus_id))
    if paper is None:
        return None
    if ctx._is_leaking(paper):
        return None
    refs = paper.get("key_references") or []
    out = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        rid = r.get("corpus_id")
        if rid is None:
            continue
        ref_paper = ctx.corpus.papers_dict.get(rid)
        if ref_paper is None:
            continue
        shaped = ctx._shape_paper(ref_paper)
        if shaped is None:
            continue
        out.append(shaped)
    return out


# ---------------------------------------------------------------------------
# Tool dispatcher used by the agent loop in baseline_agent_gpt.py
# ---------------------------------------------------------------------------

def dispatch_tool_call(tool_name: str, arguments_json: str, ctx: ToolContext) -> str:
    """Run a tool by name with JSON-encoded arguments. Returns a JSON-encoded string result."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON arguments."})

    if tool_name == "search_papers":
        result = search_papers(args.get("query", ""), ctx, limit=args.get("limit", 10))
    elif tool_name == "get_paper":
        result = get_paper(args.get("corpus_id"), ctx)
    elif tool_name == "get_author_recent_papers":
        result = get_author_recent_papers(args.get("author_id"), ctx, limit=args.get("limit", 10))
    elif tool_name == "get_references_of":
        result = get_references_of(args.get("corpus_id"), ctx)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# CorpusView and GritLMRuntime are constructed once in baseline_agent_gpt.py
# ---------------------------------------------------------------------------

class CorpusView:
    """Read-only shared corpus state (papers dict, sd2pubs, FAISS index over GRIT embeddings)."""

    def __init__(self, papers_dict: Dict, sd2publications: Dict, faiss_index, faiss_id_map: List):
        self.papers_dict = papers_dict
        self.sd2publications = sd2publications
        self.faiss_index = faiss_index
        # faiss_id_map[i] = corpus_id at FAISS row i; FAISS returns row indices.
        self.faiss_id_map = faiss_id_map


class GritLMRuntime:
    """Wraps a single GritLM model behind a thread lock for safe concurrent query embedding.

    The corpus papers are pre-embedded (loaded from .pkl). This object only embeds *queries*
    on demand, so it sees a small fraction of the embedding traffic.
    """

    def __init__(self, model_name: str = "GritLM/GritLM-7B"):
        from gritlm import GritLM
        self._model = GritLM(model_name, torch_dtype="auto", device_map="auto")
        self._lock = threading.Lock()
        # GritLM uses an instruction prefix; for retrieval-style queries:
        self._instr = "<|user|>\nGiven a topic or claim, retrieve scientific papers that are most relevant to it.\n<|embed|>\n"

    def embed_query(self, query: str) -> np.ndarray:
        with self._lock:
            emb = self._model.encode([query], instruction=self._instr)[0]
        return np.asarray(emb, dtype=np.float32).reshape(-1)
