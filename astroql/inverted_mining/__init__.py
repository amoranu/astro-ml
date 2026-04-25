"""Inverted rule mining: pick labeled charts, snapshot natal+transit at the
known event, RAG-mine the classical corpus per tradition, draft AstroQL
rules from the retrieved passages.

Pipeline modules (run in order):
  select_subjects.py  -> subjects.json  (stratified sample)
  snapshot.py         -> snapshots/{id}.json  (death-state bundle per subject)
  rag_mine.py         -> rag_results/{id}_{tradition}.json
  cluster.py          -> drafted yoga rules (audit + YAML)
"""
