import pandas as pd
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document


class CSVLoader:
    def __init__(
        self,
        path: str,
        nrows: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ):
        self.path = path
        self.nrows = nrows
        self.filters = filters or {}

    def load(self) -> List[Document]:
        df = pd.read_csv(self.path, nrows=self.nrows)

        # Apply column filters
        for col, val in self.filters.items():
            if col in df.columns:
                df = df[df[col] == val]

        docs = []
        for _, row in df.iterrows():
            if not isinstance(row.get("text"), str):
                continue

            docs.append(
                Document(
                    page_content=row["text"],
                    metadata={
                        "note_id":    str(row.get("note_id", "")),
                        "subject_id": str(row.get("subject_id", "")),
                        "hadm_id":    str(row.get("hadm_id", "")),
                        "note_type":  str(row.get("note_type", row.get("category", ""))),
                        "charttime":  str(row.get("charttime", "")),
                    },
                )
            )

        return docs
