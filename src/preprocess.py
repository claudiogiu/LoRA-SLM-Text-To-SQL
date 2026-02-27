from dataclasses import dataclass
from typing import List, Dict
from datasets import Dataset
import os
import json
import warnings

warnings.filterwarnings("ignore")

@dataclass
class WikiSQLSeq2SeqFormatter:
    """
    Interface for formatting the WikiSQL dataset into input-output pairs
    suitable for encoder-decoder (seq2seq) training workflows.

    Attributes:
        split (str): Dataset split to process ('train', 'dev', or 'test').

    Properties:
        project_root (str): Absolute path to the project root directory.
        data_path (str): Path to the directory containing WikiSQL files.
        main_filepath (str): Path to the JSONL file for the chosen split.
        tables_filepath (str): Path to the JSONL file containing table definitions.

    Methods:
        _load_tables() -> Dict[str, dict]:
            Loads table definitions from JSONL into a dictionary keyed by table_id.

        _build_sql(row: dict, table: dict) -> str:
            Constructs a human-readable SQL query from the structured representation
            of a WikiSQL row and its associated table schema.

        _parse_jsonl() -> List[Dict[str, str]]:
            Parses the JSONL file for the chosen split and returns a list of
            input-output pairs combining natural language questions with
            their corresponding SQL queries.

        to_dataset() -> Dataset:
            Converts the parsed pairs into a Hugging Face Dataset object,
            ready for seq2seq fine-tuning with question-to-SQL mappings.
    
    """

    split: str

    def __post_init__(self):
        allowed_splits = {"train", "dev", "test"}
        if self.split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{self.split}'. Allowed values are: {', '.join(allowed_splits)}"
            )

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    @property
    def data_path(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def main_filepath(self) -> str:
        return os.path.join(self.data_path, f"{self.split}.jsonl")

    @property
    def tables_filepath(self) -> str:
        return os.path.join(self.data_path, f"{self.split}.tables.jsonl")

    def _load_tables(self) -> Dict[str, dict]:
        with open(self.tables_filepath, encoding="utf-8") as f:
            tables = [json.loads(line) for line in f]
        return {t["id"]: t for t in tables}

    def _build_sql(self, row: dict, table: dict) -> str:
        agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
        cond_ops = ["=", ">", "<"]

        columns = table["header"]
        sel_col = columns[row["sql"]["sel"]]
        agg = agg_ops[row["sql"]["agg"]]
        select_part = f"SELECT {agg} {sel_col}" if agg else f"SELECT {sel_col}"

        conds = []
        for c in row["sql"]["conds"]:
            col_idx, op_idx, val = c
            conds.append(f"{columns[col_idx]} {cond_ops[op_idx]} '{val}'")
        where_part = " WHERE " + " AND ".join(conds) if conds else ""

        return f"{select_part} FROM {table['id']}{where_part};"

    def _parse_jsonl(self) -> List[Dict[str, str]]:
        tables = self._load_tables()
        pairs = []
        with open(self.main_filepath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                table = tables[row["table_id"]]
                sql_query = self._build_sql(row, table)
                input_text = (
                    f"Question: {row['question']}\n"
                    f"Table ID: {row['table_id']}\n"
                    f"Columns: {', '.join(table['header'])}"
                )
                pairs.append({"input": input_text, "output": sql_query})
        return pairs

    def to_dataset(self) -> Dataset:
        pairs = self._parse_jsonl()
        return Dataset.from_list(pairs)


@dataclass
class WikiSQLCausalLMInstructionFormatter:
    """
    Interface for formatting the WikiSQL dataset into instruction-style prompts
    suitable for decoder-only (causal LM) training workflows.

    Attributes:
        split (str): Dataset split to process ('train', 'dev', or 'test').

    Properties:
        project_root (str): Absolute path to the project root directory.
        data_path (str): Path to the directory containing WikiSQL files.
        main_filepath (str): Path to the JSONL file for the chosen split.
        tables_filepath (str): Path to the JSONL file containing table definitions.

    Methods:
        _load_tables() -> Dict[str, dict]:
            Loads table definitions from JSONL into a dictionary keyed by table_id.
        
        _build_sql(row: dict, table: dict) -> str:
            Constructs a human-readable SQL query from the structured representation
            of a WikiSQL row and its associated table schema.
        
        _parse_jsonl() -> List[Dict[str, str]]:
            Parses the JSONL file for the chosen split and returns a list of
            instruction-style prompts combining natural language questions with
            their corresponding SQL queries.
        
        to_dataset() -> Dataset:
            Converts the parsed prompts into a Hugging Face Dataset object,
            ready for causal LM fine-tuning with instruction-response pairs.
    
    """

    split: str

    def __post_init__(self):
        allowed_splits = {"train", "dev", "test"}
        if self.split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{self.split}'. Allowed values are: {', '.join(allowed_splits)}"
            )

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    @property
    def data_path(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def main_filepath(self) -> str:
        return os.path.join(self.data_path, f"{self.split}.jsonl")

    @property
    def tables_filepath(self) -> str:
        return os.path.join(self.data_path, f"{self.split}.tables.jsonl")

    def _load_tables(self) -> Dict[str, dict]:
        with open(self.tables_filepath, encoding="utf-8") as f:
            tables = [json.loads(line) for line in f]
        return {t["id"]: t for t in tables}

    def _build_sql(self, row: dict, table: dict) -> str:
        agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
        cond_ops = ["=", ">", "<"]

        columns = table["header"]
        sel_col = columns[row["sql"]["sel"]]
        agg = agg_ops[row["sql"]["agg"]]
        select_part = f"SELECT {agg} {sel_col}" if agg else f"SELECT {sel_col}"

        conds = []
        for c in row["sql"]["conds"]:
            col_idx, op_idx, val = c
            conds.append(f"{columns[col_idx]} {cond_ops[op_idx]} '{val}'")
        where_part = " WHERE " + " AND ".join(conds) if conds else ""

        return f"{select_part} FROM {table['id']}{where_part};"

    def _parse_jsonl(self) -> List[Dict[str, str]]:
        tables = self._load_tables()
        prompts = []
        with open(self.main_filepath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                table = tables[row["table_id"]]
                sql_query = self._build_sql(row, table)

                prompt_text = (
                   "Generate the SQL query that answers the question using ONLY the columns listed.\n"
                   f"Question: {row['question']}\n"
                   f"Table ID: <{row['table_id']}>\n"
                   f"Columns: {', '.join(table['header'])}\n\n"
                   "### SQL:\n"
                   f"{sql_query}"
                )
                prompts.append({"prompt": prompt_text})
        return prompts

    def to_dataset(self) -> Dataset:
        prompts = self._parse_jsonl()
        return Dataset.from_list(prompts)
    

@dataclass
class WikiSQLCausalLMChatFormatter:
    """
    Interface for formatting the WikiSQL dataset into chat-style prompts
    suitable for decoder-only (causal LM) training workflows.

    Attributes:
        split (str): Dataset split to process ('train', 'dev', or 'test').

    Properties:
        project_root (str): Absolute path to the project root directory.
        data_path (str): Path to the directory containing WikiSQL files.
        main_filepath (str): Path to the JSONL file for the chosen split.
        tables_filepath (str): Path to the JSONL file containing table definitions.

    Methods:
        _load_tables() -> Dict[str, dict]:
            Loads table definitions from JSONL into a dictionary keyed by table_id.
        
        _build_sql(row: dict, table: dict) -> str:
            Constructs a human-readable SQL query from the structured representation
            of a WikiSQL row and its associated table schema.
        
        _parse_jsonl() -> List[Dict[str, str]]:
            Parses the JSONL file for the chosen split and returns a list of
            chat-style prompts combining natural language questions with
            their corresponding SQL queries.
        
        to_dataset() -> Dataset:
            Converts the parsed prompts into a Hugging Face Dataset object,
            ready for causal LM fine-tuning with chat-style prompt representations.
    
    """

    split: str

    def __post_init__(self):
        allowed_splits = {"train", "dev", "test"}
        if self.split not in allowed_splits:
            raise ValueError(
                f"Invalid split '{self.split}'. Allowed values are: {', '.join(allowed_splits)}"
            )

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    @property
    def data_path(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def main_filepath(self) -> str:
        return os.path.join(self.data_path, f"{self.split}.jsonl")

    @property
    def tables_filepath(self) -> str:
        return os.path.join(self.data_path, f"{self.split}.tables.jsonl")

    def _load_tables(self) -> Dict[str, dict]:
        with open(self.tables_filepath, encoding="utf-8") as f:
            tables = [json.loads(line) for line in f]
        return {t["id"]: t for t in tables}

    def _build_sql(self, row: dict, table: dict) -> str:
        agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
        cond_ops = ["=", ">", "<"]

        columns = table["header"]
        sel_col = columns[row["sql"]["sel"]]
        agg = agg_ops[row["sql"]["agg"]]
        select_part = f"SELECT {agg} {sel_col}" if agg else f"SELECT {sel_col}"

        conds = []
        for c in row["sql"]["conds"]:
            col_idx, op_idx, val = c
            conds.append(f"{columns[col_idx]} {cond_ops[op_idx]} '{val}'")
        where_part = " WHERE " + " AND ".join(conds) if conds else ""

        return f"{select_part} FROM {table['id']}{where_part};"

    def _parse_jsonl(self) -> List[Dict[str, str]]:
        tables = self._load_tables()
        prompts = []
        with open(self.main_filepath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                table = tables[row["table_id"]]
                sql_query = self._build_sql(row, table)

                prompt_text = (
                    "### Instruction:\n"
                    "Generate ONLY the SQL query that answers the question using the columns provided.\n"
                    "### Input:\n"
                    f"Question: {row['question']}\n"
                    f"Table ID: <{row['table_id']}>\n"
                    f"Columns: {', '.join(table['header'])}\n\n"
                    "### Response:\n"
                    f"{sql_query}"
                )
                prompts.append({"prompt": prompt_text})
        return prompts

    def to_dataset(self) -> Dataset:
        prompts = self._parse_jsonl()
        return Dataset.from_list(prompts)