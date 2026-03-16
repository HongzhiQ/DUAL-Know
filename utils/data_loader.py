
import json
import pickle
import os
import networkx as nx
from typing import Dict, List, Optional


def load_jsonl(filepath: str) -> List[dict]:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_entity_table(filepath: str) -> Dict[str, dict]:

    entities = {}
    for item in load_jsonl(filepath):
        entities[item["id"]] = {
            "name": item["name"],
            "type": item.get("type", ""),
            "description": item.get("description", ""),
        }
    return entities


def load_kg_triples(filepath: str) -> List[dict]:

    triples = load_jsonl(filepath)
    return triples


def load_kg_graph(filepath: str) -> Optional[nx.DiGraph]:

    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            graph = pickle.load(f)
        return graph
    return None


def load_test_qa(filepath: str) -> List[dict]:

    qa = load_jsonl(filepath)
    return qa


def build_graph_from_triples(triples: List[dict], entities: Dict[str, dict]) -> nx.DiGraph:

    G = nx.DiGraph()
    for eid, einfo in entities.items():
        G.add_node(eid, name=einfo["name"], type=einfo["type"], description=einfo["description"])
    for t in triples:
        hid, tid = t["head_id"], t["tail_id"]
        if hid not in G:
            G.add_node(hid, name=t.get("head", ""), type=t.get("head_type", ""), description="")
        if tid not in G:
            G.add_node(tid, name=t.get("tail", ""), type=t.get("tail_type", ""), description="")
        G.add_edge(hid, tid, relation=t["relation"])
    return G
