import os
import sys
import copy
import pytz
import time
import json
import torch
import faiss
import pickle
import sklearn
import requests
import datetime
import logging
import multiprocessing
import numpy as np
from tqdm import tqdm
from typing import Any
from collections import defaultdict

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

logging.Formatter.converter = lambda *args: datetime.datetime.now(tz=pytz.timezone('America/Los_Angeles')).timetuple()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s: %(lineno)d: %(levelname)s: %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p")

model_tokenizer_cache = {
    "gtr": None,
    "specter2": None,
    "grit": None,
    "openai": None,
    "anthropic": None,
    "llama": None,
}



#### Logging ####

def log(message: str) -> None:
    logging.info(message)

def get_current_time():
    return datetime.datetime.now(tz=pytz.timezone('America/Los_Angeles')).strftime("%m/%d/%Y %I:%M:%S %p")



#### File Operations ####

def load_json(file_path: str) -> Any:
    logging.info(f"Reading from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            contents = json.load(f)
            metadata = contents["metadata"] if "metadata" in contents else None
            data = contents["data"] if "data" in contents else contents
    elif file_path.endswith(".jsonl"):
        with open(file_path, "r") as f:
            metadata = None
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return data, metadata

def save_json(data: Any, file_path: str, metadata = None, overwrite: bool = False) -> str:
    # if directory doesn't exist, create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        logging.info(f"Creating directory: {directory}")
        os.makedirs(directory)
    
    while os.path.exists(file_path) and not overwrite:
        logging.warning(f"File already exists: {file_path}")
        logging.info(f"Switching file path to {file_path.replace('.json', '_new.json')}")
        file_path = file_path.replace(".json", "_new.json")
        
    logging.info(f"Saving {len(data)} records to {file_path}")    
    if file_path.endswith(".json"):
        if metadata is not None:
            full_data = {"metadata": metadata, "data": data}
            data = full_data
        with open(file_path, "w") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
    elif file_path.endswith(".jsonl"):    
        with open(file_path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return file_path

def read_json(file_path: str) -> Any:
    return load_json(file_path)

def write_json(data: Any, file_path: str, metadata = None, overwrite: bool = False) -> str:
    return save_json(data, file_path, metadata, overwrite)

def update_metadata(existing_metadata, args):
    existing_metadata = copy.deepcopy(existing_metadata)
    existing_metadata.append({
        "script": sys.argv[0],
        "args": vars(args)
    })
    return existing_metadata
    
def load_pkl(file_path: str) -> Any:
    logging.info(f"Reading from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    elif file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        with open(file_path, "rb") as f:
            contents = pickle.load(f)
            if isinstance(contents, dict) and "data" in contents:
                data = contents["data"]
                metadata = contents["metadata"] if "metadata" in contents else None
            else:
                data = contents
                metadata = None
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    return data, metadata

def save_pkl(data: Any, file_path: str, metadata=None, overwrite: bool = False, protocol=pickle.HIGHEST_PROTOCOL) -> str:
    # if directory doesn't exist, create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    while os.path.exists(file_path) and not overwrite:
        logging.warning(f"File already exists: {file_path}")
        logging.info(f"Switching file path to {file_path.replace('.pkl', '_new.pkl')}")
        file_path = file_path.replace(".pkl", "_new.pkl")
        file_path = file_path.replace(".pickle", "_new.pickle")
    logging.info(f"Saving {len(data)} records to {file_path}")
    if file_path.endswith(".pkl") or file_path.endswith(".pickle"):
        with open(file_path, "wb") as f:
            pickle.dump({"metadata": metadata, "data": data}, f, protocol)
    else:
        raise ValueError(f"Unsupported save file format: {file_path}")



def _strip_none_values(obj):
    """Recursively strip None values from dicts loaded from Parquet/HuggingFace.
    Parquet requires a fixed schema so missing fields become None rather than absent keys.
    This restores the original JSON behavior where missing fields were simply absent."""
    if isinstance(obj, dict):
        return {k: _strip_none_values(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none_values(item) for item in obj]
    return obj

def load_corpus(hf_repo_id="allenai/prescience", split="test", embeddings_dir=None, embedding_type=None, load_sd2publications=True):
    """
    Load corpus data from HuggingFace Hub and optionally embeddings from local disk.

    Args:
        hf_repo_id: HuggingFace repo ID (default: "allenai/prescience")
        split: Dataset split ('train' or 'test', default: 'test')
        embeddings_dir: Local directory with embedding files (required if embedding_type specified)
        embedding_type: Embedding type to load ('gtr', 'specter2', 'grit', or None)
        load_sd2publications: Whether to load author mapping (default True)

    Returns:
        Tuple of (all_papers, sd2publications, all_embeddings)
        - all_papers: List of paper dicts
        - sd2publications: Dict mapping author_id -> list of corpus_ids (None if load_sd2publications=False)
        - all_embeddings: Dict mapping corpus_id -> embedding (None if embedding_type is None)
    """
    from huggingface_hub import hf_hub_download
    log(f"Loading corpus from HuggingFace: {hf_repo_id}")
    dataset = load_dataset(hf_repo_id)
    log(f"Using split: {split}")
    all_papers = []
    for paper in dataset[split]:
        all_papers.append(_strip_none_values(dict(paper)))
    log(f"Loaded {len(all_papers)} papers from HuggingFace")
    sd2publications = None
    if load_sd2publications:
        log("Downloading author mapping files")
        sd2pubs_path = hf_hub_download(repo_id=hf_repo_id, filename="author_publications.jsonl", repo_type="dataset")
        sd2publications = {}
        with open(sd2pubs_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                sd2publications[entry['key']] = entry['value']
        log(f"Loaded author mappings for {len(sd2publications)} authors")
    all_embeddings = None
    if embedding_type and embeddings_dir:
        embedding_path = os.path.join(embeddings_dir, f"all_papers.{embedding_type}_embeddings.pkl")
        if os.path.exists(embedding_path):
            all_embeddings, _ = load_pkl(embedding_path)
            log(f"Loaded {len(all_embeddings)} embeddings of type {embedding_type}")
        else:
            log(f"Warning: Embedding file not found: {embedding_path}")
    return all_papers, sd2publications, all_embeddings


#### Semantic Scholar API ####

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

def _get_s2_api_key():
    """Return the S2 API key from the environment, or raise an error if not set."""
    if "S2_API_KEY" not in os.environ:
        raise RuntimeError("S2_API_KEY environment variable is not set. Get an API key at https://www.semanticscholar.org/product/api#api-key")
    return os.environ["S2_API_KEY"]

def s2_batch_lookup(ids, url, fields, batch_size=500, sleep_time=1.5, num_retries=5, sleep_time_on_error=10, progress_desc="S2 batch lookup"):
    """POST batched requests to an S2 batch endpoint (e.g., /paper/batch, /author/batch). Returns list of result dicts (None for IDs not found)."""
    ids = list(set(ids))  # deduplicate
    records = []
    s2_api_key = _get_s2_api_key()
    for i in tqdm(range(0, len(ids), batch_size), desc=progress_desc):
        start_time = time.time()
        ids_batch = ids[i:i + batch_size]
        for _ in range(num_retries):
            try:
                response = requests.post(
                    url,
                    headers={"x-api-key": s2_api_key},
                    params={"fields": ",".join(fields)},
                    json={"ids": ids_batch},
                ).json()
                if not isinstance(response, list):
                    raise Exception(f"Unexpected response: {str(response)[:200]}")
                records.extend(response)
                break
            except Exception as e:
                logging.error(f"Error: {e}")
                logging.info(f"Retrying in {sleep_time_on_error} seconds")
                time.sleep(sleep_time_on_error)
        elapsed = time.time() - start_time
        time.sleep(max(0, sleep_time - elapsed))
    return records

def s2_fetch_references(corpus_ids, fields, sleep_time=1.5, num_retries=5, sleep_time_on_error=10, progress_desc="Fetching references"):
    """Fetch references for papers using per-paper API calls. Returns records in the same format as s2_batch_lookup with /paper/batch would (list of dicts with 'corpusId' and 'references' keys), so callers can use either interchangeably."""
    ref_fields = fields if "isInfluential" in fields else ["isInfluential"] + fields
    records = []
    for cid in tqdm(corpus_ids, desc=progress_desc):
        start_time = time.time()
        refs_raw = s2_get_paper(f"CorpusId:{cid}", ref_fields, endpoint_suffix="references", num_retries=num_retries, sleep_time_on_error=sleep_time_on_error)
        refs = []
        if refs_raw is not None:
            for ref in refs_raw:
                cited = ref.get("citedPaper") or {}
                cited["isInfluential"] = ref.get("isInfluential", False)
                refs.append(cited)
        records.append({"corpusId": int(cid), "references": refs})
        elapsed = time.time() - start_time
        time.sleep(max(0, sleep_time - elapsed))
    return records

def s2_get_paper(paper_id, fields, endpoint_suffix=None, num_retries=5, sleep_time_on_error=10):
    """GET a single paper from S2 API, with optional sub-endpoint and auto-pagination.

    Args:
        paper_id: e.g., "CorpusId:12345" or "ARXIV:2310.12345"
        fields: list of field names to request
        endpoint_suffix: "references", "citations", or "authors" (paginates automatically)

    Returns:
        Without endpoint_suffix: paper dict, or None if not found.
        With endpoint_suffix: list of all results across pages, or None if not found.
    """
    s2_api_key = _get_s2_api_key()
    headers = {"x-api-key": s2_api_key}
    params = {"fields": ",".join(fields)}

    if endpoint_suffix is None:
        url = f"{S2_API_BASE}/paper/{paper_id}"
        for attempt in range(num_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < num_retries - 1:
                    logging.error(f"Error fetching {url}: {e}")
                    time.sleep(sleep_time_on_error)
                else:
                    logging.error(f"Failed after {num_retries} retries for {url}: {e}")
                    return None
    else:
        url = f"{S2_API_BASE}/paper/{paper_id}/{endpoint_suffix}"
        all_results = []
        offset = 0
        while True:
            paginated_params = {**params, "offset": offset, "limit": 1000}
            for attempt in range(num_retries):
                try:
                    response = requests.get(url, headers=headers, params=paginated_params)
                    if response.status_code == 404:
                        return None
                    response.raise_for_status()
                    data = response.json()
                    all_results.extend(data.get("data", []))
                    if "next" not in data:
                        return all_results
                    offset = data["next"]
                    break
                except Exception as e:
                    if attempt < num_retries - 1:
                        logging.error(f"Error fetching {url} (offset={offset}): {e}")
                        time.sleep(sleep_time_on_error)
                    else:
                        logging.error(f"Failed after {num_retries} retries for {url} (offset={offset}): {e}")
                        return all_results if all_results else None
        return all_results

def s2_get_author(author_id, fields, endpoint_suffix=None, num_retries=5, sleep_time_on_error=10):
    """GET a single author from S2 API, with optional sub-endpoint and auto-pagination.

    Args:
        author_id: numeric S2 author ID (string)
        fields: list of field names to request
        endpoint_suffix: "papers" (paginates automatically)

    Returns:
        Without endpoint_suffix: author dict, or None if not found.
        With endpoint_suffix: list of all results across pages, or None if not found.
    """
    s2_api_key = _get_s2_api_key()
    headers = {"x-api-key": s2_api_key}
    params = {"fields": ",".join(fields)}

    if endpoint_suffix is None:
        url = f"{S2_API_BASE}/author/{author_id}"
        for attempt in range(num_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt < num_retries - 1:
                    logging.error(f"Error fetching author {author_id}: {e}")
                    time.sleep(sleep_time_on_error)
                else:
                    logging.error(f"Failed after {num_retries} retries for author {author_id}: {e}")
                    return None
    else:
        url = f"{S2_API_BASE}/author/{author_id}/{endpoint_suffix}"
        all_results = []
        offset = 0
        while True:
            paginated_params = {**params, "offset": offset, "limit": 1000}
            for attempt in range(num_retries):
                try:
                    response = requests.get(url, headers=headers, params=paginated_params)
                    if response.status_code == 404:
                        return None
                    response.raise_for_status()
                    data = response.json()
                    all_results.extend(data.get("data", []))
                    if "next" not in data:
                        return all_results
                    offset = data["next"]
                    break
                except Exception as e:
                    if attempt < num_retries - 1:
                        logging.error(f"Error fetching author {author_id}/{endpoint_suffix} (offset={offset}): {e}")
                        time.sleep(sleep_time_on_error)
                    else:
                        logging.error(f"Failed after {num_retries} retries for author {author_id}/{endpoint_suffix} (offset={offset}): {e}")
                        return all_results if all_results else None
        return all_results


#### arXiv Snapshot ####

def load_arxiv_snapshot(snapshot_path):
    """Load arXiv metadata snapshot (one JSON per line) and return dict mapping arxiv_id -> record."""
    arxiv_papers = {}
    with open(snapshot_path, "r") as f:
        for line in tqdm(f, desc="Loading arXiv snapshot"):
            record = json.loads(line)
            arxiv_papers[record["id"]] = record
    log(f"Loaded {len(arxiv_papers)} papers from arXiv snapshot")
    return arxiv_papers


#### S2 Record Processing ####

def parse_s2_paper_records(records, arxiv_snapshot=None):
    """Extract standardized paper fields from S2 API batch response records.

    Filters to arXiv-only papers with valid dates. If arxiv_snapshot is provided,
    uses it for categories and better title/abstract.

    Returns list of dicts with keys: corpus_id, arxiv_id, date, title, abstract, categories.
    """
    useful_data = []
    for record in records:
        if record is None:
            continue
        if record.get("corpusId") is None:
            continue
        external_ids = record.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id is None:
            continue
        date = record.get("publicationDate")
        if date is None:
            continue
        title = (record.get("title") or "").strip()
        abstract = (record.get("abstract") or "").strip()
        categories = []
        if arxiv_snapshot and arxiv_id in arxiv_snapshot:
            snapshot_record = arxiv_snapshot[arxiv_id]
            categories = snapshot_record["categories"].strip().split()
            title = snapshot_record["title"].strip().replace("\n", " ")
            abstract = snapshot_record["abstract"].strip().replace("\n", " ")
        useful_data.append({
            "corpus_id": str(record["corpusId"]),
            "arxiv_id": arxiv_id,
            "date": date,
            "title": title,
            "abstract": abstract,
            "categories": categories,
            "roles": [],
        })
    return useful_data



###### Retrieval ######

def aggregate_embeddings(embeddings, distance_metric):
    for i, embedding in enumerate(embeddings):
        if len(embedding.shape) > 1 and embedding.shape[0] == 1:
            embeddings[i] = embedding.flatten()
            
    if distance_metric == "l2":
        return np.mean(embeddings, axis=0)
    if distance_metric == "cosine":
        embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
        mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding / np.linalg.norm(mean_embedding)

def normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row of x."""
    x = x.astype(np.float32, copy=True)
    if x.ndim == 1:
        x = x[None, :]
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x /= norms
    return x

def create_index(embeddings_dict, distance_metric, chunk_size=2**16, use_gpu=True):
    """Build a FAISS Flat index. Returns (faiss_index, idx2id, id2indices, distance_metric, pending). Uses GPU if available and use_gpu=True."""
    if len(embeddings_dict) == 0:
        raise ValueError("embeddings_dict is empty")

    idx2id = list(embeddings_dict.keys())
    id2indices = defaultdict(list)
    for i, ext_id in enumerate(idx2id):
        id2indices[ext_id].append(i)

    embeddings = np.stack([np.asarray(embeddings_dict[_id], dtype=np.float32) for _id in idx2id], axis=0)
    dim = embeddings.shape[1]

    if distance_metric == "cosine":
        embeddings = normalize_rows(embeddings)
        metric = faiss.METRIC_INNER_PRODUCT
    elif distance_metric == "l2":
        metric = faiss.METRIC_L2
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    use_gpu = use_gpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    if metric == faiss.METRIC_L2:
        res = faiss.StandardGpuResources() if use_gpu else None
        faiss_index = faiss.GpuIndexFlatL2(res, dim) if use_gpu else faiss.IndexFlatL2(dim)
    else:
        res = faiss.StandardGpuResources() if use_gpu else None
        faiss_index = faiss.GpuIndexFlatIP(res, dim) if use_gpu else faiss.IndexFlatIP(dim)

    for i in range(0, embeddings.shape[0], chunk_size):
        faiss_index.add(embeddings[i : i + chunk_size])

    pending = {}  # id -> vec, dict so duplicates are overwritten
    return faiss_index, idx2id, id2indices, distance_metric, pending


def flush_index(index):
    """Flush pending vectors to FAISS in batch."""
    faiss_index, idx2id, id2indices, distance_metric, pending = index
    if len(pending) == 0:
        return index

    ids = list(pending.keys())
    vecs = np.stack(list(pending.values()), axis=0)
    faiss_index.add(vecs)

    start_idx = len(idx2id)
    for i, ext_id in enumerate(ids):
        idx2id.append(ext_id)
        id2indices[ext_id].append(start_idx + i)

    pending.clear()
    return faiss_index, idx2id, id2indices, distance_metric, pending

def query_index(index, query_embeddings, k, oversample=3):
    """Query the index. Returns (retrieved_ids_lists, distances_lists). Filters out tombstoned entries."""
    index = flush_index(index)
    faiss_index, idx2id, id2indices, distance_metric, pending = index
    query_embeddings = np.atleast_2d(np.asarray(query_embeddings, dtype=np.float32))

    if distance_metric == "cosine":
        query_embeddings = normalize_rows(query_embeddings)

    n_queries = query_embeddings.shape[0]
    ntotal = faiss_index.ntotal

    if ntotal == 0:
        return [[None] * k for _ in range(n_queries)], np.full((n_queries, k), np.nan, dtype=np.float32)

    # GPU indices have a max k of 2048
    is_gpu = hasattr(faiss_index, "getDevice")
    max_k = 2048 if is_gpu else ntotal

    current_oversample = oversample
    while True:
        k_search = min(max(k * current_oversample, k), ntotal, max_k)
        raw_distances, raw_indices = faiss_index.search(query_embeddings, k_search)

        out_distances = np.full((n_queries, k), np.nan, dtype=np.float32)
        retrieved_ids_lists = []
        need_retry = False

        for qi in range(n_queries):
            row_ids, row_dists = [], []
            for dist, idx in zip(raw_distances[qi], raw_indices[qi]):
                if idx < 0 or idx2id[idx] is None:
                    continue
                row_ids.append(idx2id[idx])
                row_dists.append(dist)
                if len(row_ids) >= k:
                    break

            if len(row_ids) < k and k_search < min(ntotal, max_k):
                need_retry = True

            pad_count = k - len(row_ids)
            row_ids.extend([None] * pad_count)
            row_dists.extend([np.nan] * pad_count)
            retrieved_ids_lists.append(row_ids)
            out_distances[qi, :] = np.array(row_dists, dtype=np.float32)

        if not need_retry or k_search >= min(ntotal, max_k):
            break
        current_oversample *= 2

    return retrieved_ids_lists, out_distances

def add_vector_to_index(index, new_id, new_vec):
    """Add a new (id, vec) to the pending buffer (flushed before queries)."""
    faiss_index, idx2id, id2indices, distance_metric, pending = index
    new_vec = np.asarray(new_vec, dtype=np.float32).ravel()

    if distance_metric == "cosine":
        new_vec = normalize_rows(new_vec)[0]

    pending[new_id] = new_vec
    return faiss_index, idx2id, id2indices, distance_metric, pending

def remove_vector_from_index(index, id):
    """Logically remove a vector by tombstoning (setting idx2id entry to None)."""
    faiss_index, idx2id, id2indices, distance_metric, pending = index
    for i in id2indices.get(id, []):
        idx2id[i] = None
    id2indices.pop(id, None)
    return faiss_index, idx2id, id2indices, distance_metric, pending

def replace_vector_in_index(index, id, new_vec):
    """Replace a vector by tombstoning old entries and adding the new vector."""
    index = remove_vector_from_index(index, id)
    index = add_vector_to_index(index, id, new_vec)
    return index


def calculate_ndcg(retrieved_ids, relevant_ids):
    """
    Calculate NDCG for a single query.
    """
    if len(retrieved_ids) == 0:
        log("No retrieved IDs")
        return 0
    elif len(relevant_ids) == 0:
        log("No relevant IDs")
        return 0
    elif len(retrieved_ids) < len(relevant_ids):
        log("Number of retrieved IDs is less than number of relevant IDs")
        return 0
    else:
        relevant_ids_set = set(relevant_ids)
        # Create a relevance score for each retrieved ID
        relevance_scores = [(1 if r in relevant_ids_set else 0) for r in retrieved_ids]
        
        dcg = 0     # Calculate DCG
        for i, score in enumerate(relevance_scores):
            dcg += score / np.log2(i + 2)
        
        idcg = 0    # Calculate IDCG
        for i in range(len(relevant_ids)):
            idcg += 1 / np.log2(i + 2)
        
        ndcg = dcg / idcg   # Calculate NDCG
        return ndcg

def calculate_precision_recall_f1(retrieved_ids, relevant_ids):
    """
    Calculate Precision, Recall, and F1-score for a single query.
    """
    if len(retrieved_ids) == 0:
        log("No retrieved IDs")
        return 0, 0, 0
    elif len(relevant_ids) == 0:
        log("No relevant IDs")
        return 0, 0, 0
    
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)
    
    true_positives = len(retrieved_set & relevant_set)
    false_positives = len(retrieved_set - relevant_set)
    false_negatives = len(relevant_set - retrieved_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score



##### Data Processing #####
def get_intersection(list_of_lists):
    intersection = None
    for l in list_of_lists:
        if intersection is None:
            intersection = set([paper["corpus_id"] for paper in l])
        else:
            intersection = intersection & set([paper["corpus_id"] for paper in l])
        
    for i, l in enumerate(list_of_lists):
        list_of_lists[i] = [paper for paper in l if paper["corpus_id"] in intersection]
        
    return list_of_lists

def shortest_common_supersequence(list1, list2):
    # Step 1: Compute the LCS (Longest Common Subsequence)
    m, n = len(list1), len(list2)
    dp = [[[] for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if list1[i] == list2[j]:
                dp[i + 1][j + 1] = dp[i][j] + [list1[i]]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], key=len)

    lcs = dp[m][n]

    # Step 2: Build the SCS from LCS
    i = j = 0
    result = []

    for c in lcs:
        while i < len(list1) and list1[i] != c:
            result.append(list1[i])
            i += 1
        while j < len(list2) and list2[j] != c:
            result.append(list2[j])
            j += 1
        result.append(c)
        i += 1
        j += 1

    # Append remaining parts
    result.extend(list1[i:])
    result.extend(list2[j:])
    return result

def filter_by_roles(records, roles):
    roles_set = set(roles)
    filtered_records = []
    for record in records:
        record_roles = record["roles"]
        if any(role in roles_set for role in record_roles):
            filtered_records.append(copy.deepcopy(record))
    return filtered_records

def add_role(records, role):
    records = copy.deepcopy(list(records))
    for record in records:
        if "roles" not in record:
            record["roles"] = []
        record["roles"] = sorted(list(set(record["roles"] + [role])))
    return records

def union_records(records1, records2):
    # Accept both dict and list inputs; return same type as first input
    input_was_dict = isinstance(records1, dict)
    if input_was_dict:
        list1_dict = records1
    else:
        list1_dict = {paper["corpus_id"]: paper for paper in records1}
    if isinstance(records2, dict):
        list2_dict = records2
    else:
        list2_dict = {paper["corpus_id"]: paper for paper in records2}
    merged_paper_set_dict = copy.deepcopy(list1_dict)
    for corpus_id, paper in tqdm(list2_dict.items(), desc="Merging record lists"):
        if corpus_id in merged_paper_set_dict:
            for key in paper:
                if key in merged_paper_set_dict[corpus_id]:
                    if paper[key] != merged_paper_set_dict[corpus_id][key]:
                        if isinstance(paper[key], list):
                            l1 = merged_paper_set_dict[corpus_id][key]
                            l2 = paper[key]
                            # handle the special case where they're equally long lists of dictionaries
                            if len(l1) > 0 and len(l1) == len(l2) and \
                                isinstance(l1[0], dict) and isinstance(l2[0], dict) and \
                                (all(l1e.items() <= l2e.items() for l1e, l2e in zip(l1, l2)) or \
                                    all(l2e.items() <= l1e.items() for l1e, l2e in zip(l1, l2))):
                                
                                # merge each pair of dictionaries
                                merged_paper_set_dict[corpus_id][key] = []
                                for l1e, l2e in zip(l1, l2):
                                    merged_paper_set_dict[corpus_id][key].append({**l1e, **l2e})
                            else:
                                # preserve the order of the list: compute shortest common supersequence
                                merged_paper_set_dict[corpus_id][key] = shortest_common_supersequence(l1, l2)
                        elif isinstance(paper[key], str):
                            # pick the longer string
                            if len(merged_paper_set_dict[corpus_id][key]) < len(paper[key]):
                                merged_paper_set_dict[corpus_id][key] = paper[key]
                        else:
                            raise ValueError(f"Merge not supported for distinct values of type {type(paper[key])} (key: {key})")
                else:
                    merged_paper_set_dict[corpus_id][key] = paper[key]
        else:
            merged_paper_set_dict[corpus_id] = paper

    if input_was_dict:
        return merged_paper_set_dict
    else:
        merged_paper_list = list(merged_paper_set_dict.values())
        merged_paper_list = sorted(merged_paper_list, key=lambda x: x["date"])
        return merged_paper_list

def union_records_multiway(list_of_lists):
    list_of_lists = copy.deepcopy(list_of_lists)
    merged_list = list_of_lists[0]
    for i in range(1, len(list_of_lists)):
        merged_list = union_records(merged_list, list_of_lists[i])
    return merged_list

def remove_unreachable_papers(all_papers):
    # Accept both dict and list inputs; return same type as input
    input_was_dict = isinstance(all_papers, dict)
    if input_was_dict:
        all_papers_dict = all_papers
    else:
        all_papers_dict = {p["corpus_id"]: p for p in all_papers}

    present_roles = set()
    target_cids = []

    for paper in tqdm(all_papers_dict.values(), desc="Preparing papers", leave=False):
        roles = paper.get("roles", [])
        present_roles.update(roles)
        if "target" in roles:
            target_cids.append(paper["corpus_id"])

    retained_corpus_ids = set()
    revised_roles_dict: dict[str, set[str]] = defaultdict(set)

    def assign_role(corpus_id: str, role: str) -> None:
        if corpus_id not in all_papers_dict:
            return
        revised_roles_dict[corpus_id].add(role)
        retained_corpus_ids.add(corpus_id)

    if "target" in present_roles:
        for cid in tqdm(target_cids, desc="Marking targets", leave=False):
            assign_role(cid, "target")

        if "target.key_reference" in present_roles:
            target_key_reference_ids = {
                ref.get("corpus_id")
                for cid in target_cids
                for ref in all_papers_dict[cid].get("key_references", [])
                if isinstance(ref, dict) and ref.get("corpus_id") is not None
            }
            for cid in tqdm(sorted(target_key_reference_ids), desc="Marking target key refs", leave=False):
                assign_role(cid, "target.key_reference")

        if "target.all_references" in present_roles:
            target_all_reference_ids = {
                ref.get("corpus_id")
                for cid in target_cids
                for ref in all_papers_dict[cid].get("all_references", [])
                if isinstance(ref, dict) and ref.get("corpus_id") is not None
            }
            for cid in tqdm(sorted(target_all_reference_ids), desc="Marking target all refs", leave=False):
                assign_role(cid, "target.all_references")

        if "target.author.publication_history" in present_roles:
            target_auth_pub_hist_cids = {
                history_id
                for cid in target_cids
                for author in all_papers_dict[cid].get("authors", [])
                for history_id in author.get("publication_history", [])
                if history_id is not None
            }
            for cid in tqdm(sorted(target_auth_pub_hist_cids), desc="Marking pub histories", leave=False):
                assign_role(cid, "target.author.publication_history")

            if "target.author.publication_history.key_reference" in present_roles:
                target_auth_pub_hist_hi_ref_cids = {
                    ref.get("corpus_id")
                    for cid in target_auth_pub_hist_cids
                    for ref in all_papers_dict.get(cid, {}).get("key_references", [])
                    if isinstance(ref, dict) and ref.get("corpus_id") is not None
                }
                for cid in tqdm(sorted(target_auth_pub_hist_hi_ref_cids), desc="Marking pub-history key refs", leave=False):
                    assign_role(cid, "target.author.publication_history.key_reference")

            if "target.author.publication_history.all_references" in present_roles:
                target_auth_pub_hist_all_ref_cids = {
                    ref.get("corpus_id")
                    for cid in target_auth_pub_hist_cids
                    for ref in all_papers_dict.get(cid, {}).get("all_references", [])
                    if isinstance(ref, dict) and ref.get("corpus_id") is not None
                }
                for cid in tqdm(sorted(target_auth_pub_hist_all_ref_cids), desc="Marking pub-history all refs", leave=False):
                    assign_role(cid, "target.author.publication_history.all_references")

    retained_papers_dict = {}
    for cid in retained_corpus_ids:
        paper = all_papers_dict[cid]
        paper["roles"] = sorted(revised_roles_dict[cid])
        retained_papers_dict[cid] = paper

    if input_was_dict:
        return retained_papers_dict
    else:
        return list(retained_papers_dict.values())

def refine_key_references_and_publication_histories_to_ensure_membership(all_papers):
    """Iterate through all key_references and publication histories listings and remove entries missing from all_papers."""
    input_was_dict = isinstance(all_papers, dict)
    if input_was_dict:
        all_papers_dict = all_papers
    else:
        all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}

    total_key_references_removed = 0
    total_publication_history_entries_removed = 0
    for paper in tqdm(all_papers_dict.values(), desc="Refining key references and publication histories"):
        if "key_references" in paper:
            total_key_references_removed += len(paper["key_references"])
            paper["key_references"] = [ref for ref in paper["key_references"] if ref["corpus_id"] in all_papers_dict]
            total_key_references_removed -= len(paper["key_references"])
        if "target" in paper["roles"] and "authors" in paper:
            for author in paper["authors"]:
                if "publication_history" in author:
                    total_publication_history_entries_removed += len(author["publication_history"])
                    author["publication_history"] = [pub for pub in author["publication_history"] if pub in all_papers_dict]
                    total_publication_history_entries_removed -= len(author["publication_history"])
    log(f"Removed {total_key_references_removed} key references entries missing from all_papers")
    log(f"Removed {total_publication_history_entries_removed} publication history entries missing from all_papers")

    if input_was_dict:
        return all_papers_dict
    else:
        return list(all_papers_dict.values())

def cleanup_after_pruning(all_papers):
    """Refine references, remove targets with zero key_references, and remove unreachable papers.

    This function handles the common cleanup cycle needed after pruning papers,
    iterating until a fixed point is reached:
    1. Refine key_references and publication_histories to ensure all referenced papers exist
    2. Remove target papers that now have zero key_references
    3. Remove unreachable papers
    4. Repeat until no changes occur
    """
    input_was_dict = isinstance(all_papers, dict)
    if input_was_dict:
        all_papers_dict = all_papers
    else:
        all_papers_dict = {paper["corpus_id"]: paper for paper in all_papers}

    size_before = len(all_papers_dict) + 1  # ensure at least one iteration
    while len(all_papers_dict) < size_before:
        size_before = len(all_papers_dict)

        all_papers_dict = refine_key_references_and_publication_histories_to_ensure_membership(all_papers_dict)

        # Remove target papers with zero key_references
        target_corpus_ids = {corpus_id for corpus_id, p in all_papers_dict.items() if "target" in p["roles"]}
        zero_ref_targets = {corpus_id for corpus_id in target_corpus_ids if len(all_papers_dict[corpus_id].get("key_references", [])) == 0}
        if zero_ref_targets:
            all_papers_dict = {corpus_id: paper for corpus_id, paper in all_papers_dict.items() if corpus_id not in zero_ref_targets}
            log(f"Pruned {len(zero_ref_targets)} target papers with zero key references ({len(target_corpus_ids) - len(zero_ref_targets)} targets remain)")

        all_papers_dict = remove_unreachable_papers(all_papers_dict)

    if input_was_dict:
        return all_papers_dict
    else:
        return list(all_papers_dict.values())


def realign_roles(all_papers_dict):
    """Recompute target/key-reference/publication-history roles and return updated dict."""
    target_ids = set()
    for paper in all_papers_dict.values():
        roles = paper["roles"] if "roles" in paper else []
        if "target" in roles:
            target_ids.add(paper["corpus_id"])
    target_key_reference_ids = set()
    for cid in target_ids:
        key_refs = all_papers_dict[cid]["key_references"] if "key_references" in all_papers_dict[cid] else []
        for ref in key_refs:
            if isinstance(ref, dict) and ("corpus_id" in ref) and (ref["corpus_id"] in all_papers_dict):
                target_key_reference_ids.add(ref["corpus_id"])

    pub_history_ids = set()
    for cid in target_ids:
        authors = all_papers_dict[cid]["authors"] if "authors" in all_papers_dict[cid] else []
        for author in authors:
            publication_history = author["publication_history"] if ("publication_history" in author and author["publication_history"]) else []
            for history_cid in publication_history:
                if history_cid in all_papers_dict:
                    pub_history_ids.add(history_cid)

    pub_history_key_reference_ids = set()
    for cid in pub_history_ids:
        key_refs = all_papers_dict[cid]["key_references"] if "key_references" in all_papers_dict[cid] else []
        for ref in key_refs:
            if isinstance(ref, dict) and ("corpus_id" in ref) and (ref["corpus_id"] in all_papers_dict):
                pub_history_key_reference_ids.add(ref["corpus_id"])

    roles_to_refresh = [
        ("target.key_reference", target_key_reference_ids),
        ("target.author.publication_history", pub_history_ids),
        ("target.author.publication_history.key_reference", pub_history_key_reference_ids),
    ]
    for paper in all_papers_dict.values():
        roles = set(paper["roles"]) if "roles" in paper else set()
        for role, valid_ids in roles_to_refresh:
            roles.discard(role)
            if paper["corpus_id"] in valid_ids:
                roles.add(role)
        paper["roles"] = sorted(roles)
    log(
        "Assigned roles to "
        f"{len(target_ids)} targets, "
        f"{len(target_key_reference_ids)} target key references, "
        f"{len(pub_history_ids)} publication-history papers, "
        f"{len(pub_history_key_reference_ids)} publication-history key references"
    )
    return all_papers_dict




##### Prompting #####

def get_title_abstract_string(record):
    title = record["title"]
    abstract = record["abstract"]
    return f"Title: {title}\nAbstract: {abstract}"

def embed_on_gpus(document_list, embedding_type, cache_model=True, return_dict=True, quiet=False):
    """
    Embed a list of documents on available GPUs.
    """
    model, tokenizer = None, None
    if cache_model and (embedding_type in model_tokenizer_cache) and (model_tokenizer_cache[embedding_type] is not None):
        model, tokenizer = model_tokenizer_cache[embedding_type]
    
    if embedding_type == "gtr":
        if model is None:        
            model = SentenceTransformer("sentence-transformers/gtr-t5-large", device="cuda")
            log("Caching GTR model")
            model_tokenizer_cache[embedding_type] = (model, None)
        embeddings = model.encode(document_list, batch_size=128).astype(np.float16)
        key_embeddings = copy.deepcopy(embeddings)
        query_embeddings = copy.deepcopy(embeddings)

    elif embedding_type == "specter2":
        from adapters import AutoAdapterModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
            model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
            model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
            model.eval(); model = model.to(device)
            log("Caching Specter2 model and tokenizer")
            model_tokenizer_cache[embedding_type] = (model, tokenizer)
        
        document_list = [
                document.split("\n")[0].replace("Title:", "").strip() + \
                tokenizer.sep_token + \
                document.split("\n")[1].replace("Abstract:", "").strip()
            if "Title:" in document else document
            for document in document_list]
        batch_size = 64
        chunks = []
        with torch.no_grad():
            for i in tqdm(range(0, len(document_list), batch_size), desc="Batches", disable=quiet):
                inputs = tokenizer(document_list[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                chunks.append(model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy().astype(np.float16))
        embeddings = np.concatenate(chunks, axis=0)
        key_embeddings = copy.deepcopy(embeddings)
        query_embeddings = copy.deepcopy(embeddings)
        
    elif embedding_type == "grit":
        if model is None:
            from gritlm import GritLM
            model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", device_map="auto", mode="embedding")
            log("Caching GritLM model")
            model_tokenizer_cache[embedding_type] = (model, None)
        key_instruction = "<|embed|>\n"
        query_instruction = "<|user|>\nGiven the title and abstract of a research paper, retrieve the title and abstract of a related research paper\n<|embed|>\n"
        key_embeddings = model.encode(document_list, batch_size=128, instruction=key_instruction, show_progress_bar=True).astype(np.float16)
        query_embeddings = model.encode(document_list, batch_size=128, instruction=query_instruction, show_progress_bar=True).astype(np.float16)
    
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    if return_dict:
        embedding_dicts_list = []
        for i in tqdm(range(len(document_list)), desc="Creating embeddings_dict_list", disable=quiet):
            embedding_dicts_list.append({
                "key": key_embeddings[i].reshape(1, -1),
                "query": query_embeddings[i].reshape(1, -1)
            })
        return embedding_dicts_list
    else:
        return key_embeddings, query_embeddings

def _embed_on_gpus_parallel(document_list, return_dict, process_id, embedding_type, assigned_gpus=None):
    if assigned_gpus is None:
        assigned_gpus = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))
    return_dict[process_id] = embed_on_gpus(document_list, embedding_type, cache_model=False, return_dict=False)

def embed_on_gpus_parallel(document_list, embedding_type, quiet=False):
    if len(document_list) == 0:
        return []

    multiprocessing.set_start_method("fork", force=True)
    num_total_gpus = torch.cuda.device_count()
    assert num_total_gpus > 0, f"{embedding_type.upper()} embeddings require GPUs"

    if embedding_type == "gtr":
        gpus_per_process = 1
    elif embedding_type == "specter2":
        gpus_per_process = 1
    elif embedding_type == "grit":
        gpus_per_process = 2

    if num_total_gpus < gpus_per_process:
        log(
            f"Insufficient GPUs ({num_total_gpus}) for parallel {embedding_type.upper()} embedding "
            f"(requires {gpus_per_process} per process); falling back to single-process embedding."
        )
        return embed_on_gpus(document_list, embedding_type, cache_model=True, return_dict=True, quiet=quiet)
    
    num_processes = num_total_gpus // gpus_per_process
    if num_processes <= 1:
        return embed_on_gpus(document_list, embedding_type, cache_model=True, return_dict=True, quiet=quiet)

    shards = [list(document_list)[i::num_processes] for i in range(num_processes)]
    assigned_gpus = [list(range(i*gpus_per_process, (i+1)*gpus_per_process)) for i in range(num_processes)]
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=_embed_on_gpus_parallel, args=(shards[i], return_dict, i, embedding_type, assigned_gpus[i]))
        p.start(); processes.append(p)
    for p in processes:
        p.join()

    key_embeddings_all = np.concatenate([return_dict[p_id][0] for p_id in sorted(return_dict)], axis=0)
    query_embeddings_all = np.concatenate([return_dict[p_id][1] for p_id in sorted(return_dict)], axis=0)

    embedding_dicts_list = []
    if embedding_type in ["gtr", "grit"]:
        for i in tqdm(range(len(document_list)), desc="Creating embeddings_dict_list", disable=quiet):
            embedding_dicts_list.append({
                "key": sklearn.preprocessing.normalize(key_embeddings_all[i].reshape(1,-1)),
                "query": sklearn.preprocessing.normalize(query_embeddings_all[i].reshape(1,-1))
            })
    else:
        for i in tqdm(range(len(document_list)), desc="Creating embeddings_dict_list", disable=quiet):
            embedding_dicts_list.append({
                "key": key_embeddings_all[i].reshape(1,-1),
                "query": query_embeddings_all[i].reshape(1,-1)
            })
    return embedding_dicts_list
