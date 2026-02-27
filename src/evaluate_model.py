import os
import asyncio
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

from preprocess import (
    WikiSQLSeq2SeqFormatter,
    WikiSQLCausalLMInstructionFormatter,
    WikiSQLCausalLMChatFormatter
)

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

warnings.filterwarnings("ignore")

LLM_JUDGE_PROMPT: str = """
You are an impartial SQL semantic equivalence judge.

Your task is to determine whether a predicted SQL query is semantically equivalent to a reference SQL query.
Your evaluation must focus strictly on meaning, not surface form.

The following differences MUST be ignored:
- whitespace, capitalization, punctuation, backticks, code fences, markdown, trailing semicolons
- different ordering of conditions in the WHERE clause
- different ordering of selected columns
- harmless differences in spacing or parentheses

The following differences MUST be treated as SEMANTICALLY SIGNIFICANT:
- missing filters, conditions, or constraints present in the reference query
- additional filters or constraints not present in the reference query
- changes to comparison operators (<, >, =, !=, <=, >=)
- changes to logical structure (AND/OR)
- changes to aggregation functions (e.g., SUM, COUNT, MAX, MIN, AVG)
- selecting a different column or a different aggregation target
- any modification that would alter the result set for any valid database instance

A prediction is GOOD (true) only if ALL conditions are satisfied:
1. Every filter, condition, and aggregation in the reference query is preserved in meaning.
2. No filter, condition, or aggregation is missing, altered, or replaced.
3. The predicted query would return the same result set as the reference query for all valid databases.
4. No semantic information from the reference query is lost or changed.

A prediction is BAD (false) if ANY of the following occur:
- a required condition is missing
- a condition is added or changed in meaning
- an aggregation is missing or different
- the selected column or aggregation target differs
- the predicted query would return a different result set

You must output ONLY one token: `true` or `false`.

Now evaluate the following pair:

PREDICTED QUERY:
{predicted_sql}

REFERENCE QUERY:
{gold_sql}

Your answer (true/false):
"""

@dataclass
class WikiSQLT5Evaluator:
    """
    Interface for evaluating T5-family encoder–decoder models fine‑tuned on the
    WikiSQL dataset, using exact-match logical‑form comparison between predicted
    and reference SQL queries.

    Attributes:
        device (torch.device): Computational device selected at initialization.
        tokenizer (AutoTokenizer): Tokenizer restored from the local T5 checkpoint.
        model (AutoModelForSeq2SeqLM): LoRA‑adapted T5 model loaded in FP16 mode.

    Methods:
        _load_dataset(split: str) -> Dataset:
            Loads and formats the specified WikiSQL split ('train', 'dev', 'test')
            using the seq2seq formatter.

        evaluate_split(split: str, max_samples: int | None, batch_size: int) -> List[Dict[str, Any]]:
            Generates SQL predictions for the selected split, computes exact-match
            logical‑form accuracy, and returns record‑level structures containing
            predicted and reference SQL queries for downstream evaluation.
    
    """

    device: torch.device = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)
    model: AutoModelForSeq2SeqLM = field(init=False)

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    @property
    def data_path(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def models_path(self) -> str:
        return os.path.join(self.project_root, "models", "T5")

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.models_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.models_path,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def _load_dataset(self, split: str) -> Dataset:
        processor = WikiSQLSeq2SeqFormatter(split=split)
        return processor.to_dataset()

    def evaluate_split(self, split: str, max_samples: int = None, batch_size: int = 32) -> None:
        dataset = self._load_dataset(split)
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

        correct_exact = 0
        all_records = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_inputs = dataset[start:end]["input"]
            gold_sqls = dataset[start:end]["output"]

            inputs = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            pred_sqls = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for pred, gold in zip(pred_sqls, gold_sqls):
                if pred.strip() == gold.strip():
                    correct_exact += 1

            for i, (pred, gold) in enumerate(zip(pred_sqls, gold_sqls)):
                all_records.append({
                    "index": start + i,
                    "predicted_sql": pred,
                    "gold_sql": gold
                })

        accuracy = correct_exact / total * 100
        print(f"Logical Form Accuracy: {accuracy:.2f}% ({correct_exact}/{total})")

        return all_records
        

@dataclass
class WikiSQLQwenEvaluator:
    """
    Interface for evaluating Qwen-family decoder‑only models fine‑tuned on the
    WikiSQL dataset, using exact-match logical‑form comparison between predicted
    and reference SQL queries.

    Attributes:
        device (torch.device): Computational device selected at initialization.
        tokenizer (AutoTokenizer): Tokenizer restored from the local Qwen checkpoint,
            configured for left‑padding and EOS‑based padding.
        model (AutoModelForCausalLM): LoRA‑adapted Qwen model loaded in FP16 mode.

    Methods:
        _load_dataset(split: str) -> Dataset:
            Loads and formats the specified WikiSQL split ('train', 'dev', 'test')
            using the instruction‑style causal‑LM formatter.

        _truncate_prompt(full_prompt: str) -> str:
            Ensures that generation begins immediately after the SQL marker by
            truncating the prompt at the '### SQL:' delimiter, preserving only the
            prefix required for continuation‑based decoding.

        evaluate_split(split: str, max_samples: int | None, batch_size: int) -> List[Dict[str, Any]]:
            Generates SQL predictions for the selected split, computes exact‑match
            logical‑form accuracy, and returns record‑level structures containing
            predicted and reference SQL queries for downstream evaluation.
    
    """

    device: torch.device = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)
    model: AutoModelForCausalLM = field(init=False)

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    @property
    def data_path(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def models_path(self) -> str:
        return os.path.join(self.project_root, "models", "Qwen")

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.models_path, fix_mistral_regex=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.models_path,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.config.use_cache = False
        self.model.eval()

    def _load_dataset(self, split: str) -> Dataset:
        processor = WikiSQLCausalLMInstructionFormatter(split=split)
        return processor.to_dataset()

    def _truncate_prompt(self, full_prompt: str) -> str:
        if "### SQL:" in full_prompt:
            return full_prompt.split("### SQL:")[0] + "### SQL:\n"
        return full_prompt

    def evaluate_split(self, split: str, max_samples: int = None, batch_size: int = 48) -> None:
        dataset = self._load_dataset(split)
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

        correct_exact = 0
        all_records = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_prompts = dataset[start:end]["prompt"]

            truncated_prompts = [self._truncate_prompt(p) for p in batch_prompts]
            gold_sqls = [p.split('### SQL:')[1].strip() for p in batch_prompts]

            inputs = self.tokenizer(
                truncated_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False
                )

            pred_sqls = []
            for text in self.tokenizer.batch_decode(outputs, skip_special_tokens=True):
                if "### SQL:" in text:
                    pred_sqls.append(text.split("### SQL:")[-1].strip())
                else:
                    pred_sqls.append(text.strip())

            for pred, gold in zip(pred_sqls, gold_sqls):
                if pred.strip() == gold.strip():
                    correct_exact += 1

            for i, (pred, gold) in enumerate(zip(pred_sqls, gold_sqls)):
                all_records.append({
                    "index": start + i,
                    "predicted_sql": pred,
                    "gold_sql": gold
                })

        accuracy = correct_exact / total * 100
        print(f"Logical Form Accuracy: {accuracy:.2f}% ({correct_exact}/{total})")

        return all_records


@dataclass
class WikiSQLLlamaEvaluator:
    """
    Interface for evaluating Llama-family decoder‑only models fine‑tuned on the
    WikiSQL dataset, using exact-match logical‑form comparison between predicted
    and reference SQL queries.

    Attributes:
        device (torch.device): Computational device selected at initialization.
        tokenizer (AutoTokenizer): Tokenizer restored from the local Llama checkpoint,
            configured for left‑padding and EOS‑based padding.
        model (AutoModelForCausalLM): LoRA‑adapted Llama model loaded in FP16 mode.

    Methods:
        _load_dataset(split: str) -> Dataset:
            Loads and formats the specified WikiSQL split ('train', 'dev', 'test')
            using the chat‑style causal‑LM formatter.

        _truncate_prompt(full_prompt: str) -> str:
            Ensures that generation begins immediately after the SQL marker by
            truncating the prompt at the '### Response:' delimiter, preserving only the
            prefix required for continuation‑based decoding.

        evaluate_split(split: str, max_samples: int | None, batch_size: int) -> List[Dict[str, Any]]:
            Generates SQL predictions for the selected split, computes exact‑match
            logical‑form accuracy, and returns record‑level structures containing
            predicted and reference SQL queries for downstream evaluation.
    
    """

    device: torch.device = field(init=False)
    tokenizer: AutoTokenizer = field(init=False)
    model: AutoModelForCausalLM = field(init=False)

    @property
    def project_root(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    @property
    def data_path(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def models_path(self) -> str:
        return os.path.join(self.project_root, "models", "Llama")

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.models_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.models_path,
            dtype=torch.float16,
            device_map="auto"
        )
        self.model.config.use_cache = False
        self.model.eval()

    def _load_dataset(self, split: str) -> Dataset:
        processor = WikiSQLCausalLMChatFormatter(split=split)
        return processor.to_dataset()

    def _truncate_prompt(self, full_prompt: str) -> str:
        if "### Response:" in full_prompt:
            return full_prompt.split("### Response:")[0] + "### Response:\n"
        return full_prompt

    def evaluate_split(self, split: str, max_samples: int = None, batch_size: int = 48) -> None:
        dataset = self._load_dataset(split)
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

        correct_exact = 0
        all_records = []

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)

            batch_prompts = dataset[start:end]["prompt"]

            truncated_prompts = [self._truncate_prompt(p) for p in batch_prompts]

            gold_sqls = [p.split("### Response:")[1].strip() for p in batch_prompts]

            inputs = self.tokenizer(
                truncated_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False
                )

            pred_sqls = []
            for text in self.tokenizer.batch_decode(outputs, skip_special_tokens=True):
                if "### Response:" in text:
                    pred_sqls.append(text.split("### Response:")[-1].strip())
                else:
                    pred_sqls.append(text.strip())

            for pred, gold in zip(pred_sqls, gold_sqls):
                if pred.strip() == gold.strip():
                    correct_exact += 1

            for i, (pred, gold) in enumerate(zip(pred_sqls, gold_sqls)):
                all_records.append({
                    "index": start + i,
                    "predicted_sql": pred,
                    "gold_sql": gold
                })

        accuracy = correct_exact / total * 100
        print(f"Logical Form Accuracy: {accuracy:.2f}% ({correct_exact}/{total})")

        return all_records


@dataclass
class WikiSQLLLMJudgeEvaluator:
    """
    Interface for assessing SQL predictions through an external large‑language‑model
    judge, applying equivalence criteria to determine whether generated SQL matches
    the reference logical form under a strict evaluation protocol.

    Attributes:
        deployment (str): Name of the Azure OpenAI deployment used for LLM‑based judging.
        api_version (str): API version for Azure OpenAI access.
        batch_size (int): Number of SQL pairs evaluated concurrently during batched judging.

    Methods:
        _call_model(prompt: str) -> str:
            Issues a retry‑protected request to the LLM judge and returns the model's
            binary verdict (“true” or “false”) for a single SQL comparison prompt.

        _judge_pair(predicted: str, gold: str) -> bool:
            Formats an equivalence‑checking prompt using the LLM_JUDGE_PROMPT template
            and obtains the LLM's verdict for a single predicted/reference pair.

        _judge_batch(batch: List[Dict[str, Any]]) -> List[bool]:
            Evaluates a batch of SQL pairs concurrently and returns a list of boolean
            judgments aligned with the batch ordering.

        _judge(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            Applies LLM‑based judging to all non‑identical SQL pairs while marking
            exact string matches as correct, returning an augmented record list
            containing the LLM's verdict for each item.

        evaluate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
            Executes the complete LLM‑as‑a‑Judge evaluation pipeline, computes the
            overall equivalence accuracy, and returns both the annotated record set
            and the aggregate accuracy metric.
    
    """

    deployment: str = "gpt-4.1"
    api_version: str = "2024-12-01-preview"
    batch_size: int = 32

    def __post_init__(self) -> None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not endpoint or not api_key:
            raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY")

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        reraise=True
    )
    async def _call_model(self, prompt: str) -> str:
        def sync_call():
            return self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a strict SQL evaluation judge."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=128,
                model=self.deployment
            )

        response = await asyncio.to_thread(sync_call)
        return (response.choices[0].message.content or "").strip().lower()

    async def _judge_pair(self, predicted: str, gold: str) -> bool:
        prompt = LLM_JUDGE_PROMPT.format(predicted_sql=predicted, gold_sql=gold)
        output = await self._call_model(prompt)
        return "true" in output

    async def _judge_batch(self, batch: List[Dict[str, Any]]) -> List[bool]:
        tasks = [
            self._judge_pair(item["predicted_sql"], item["gold_sql"])
            for item in batch
        ]
        return await asyncio.gather(*tasks)

    async def _judge(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        perfect_matches = []
        mismatches = []

        for item in records:
            if item["predicted_sql"].strip() == item["gold_sql"].strip():
                perfect_matches.append(item)
            else:
                mismatches.append(item)

        for item in perfect_matches:
            item["judge"] = True
            results.append(item)

        total = len(mismatches)

        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch = mismatches[start:end]

            judgments = await self._judge_batch(batch)

            for item, judge_value in zip(batch, judgments):
                item["judge"] = judge_value
                results.append(item)

        return results
    
    async def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        judged_records = await self._judge(records)

        total = len(judged_records)
        true_count = sum(1 for r in judged_records if r["judge"])
        accuracy = true_count / total * 100 if total > 0 else 0.0

        print(f"LLM-as-a-Judge Accuracy: {accuracy:.2f}% ({true_count}/{total})")

        return {
            "records": judged_records,
            "accuracy": accuracy
        }


if __name__ == "__main__":
    print("\nT5-Base — Logical Form Accuracy Evaluation\n")
    t5_eval: WikiSQLT5Evaluator = WikiSQLT5Evaluator()
    t5_records: List[Dict[str, str]] = t5_eval.evaluate_split("test")

    print("\nQwen2.5-3B-Instruct — Logical Form Accuracy Evaluation\n")
    qwen_eval: WikiSQLQwenEvaluator = WikiSQLQwenEvaluator()
    qwen_records: List[Dict[str, str]] = qwen_eval.evaluate_split("test")

    print("\nLLaMA-3.2-3B — Logical Form Accuracy Evaluation\n")
    llama_eval: WikiSQLLlamaEvaluator = WikiSQLLlamaEvaluator()
    llama_records: List[Dict[str, str]] = llama_eval.evaluate_split("test")

    judge: WikiSQLLLMJudgeEvaluator = WikiSQLLLMJudgeEvaluator()

    print("\nT5-Base — LLM-as-a-Judge Evaluation\n")
    t5_judged: Dict[str, Any] = asyncio.run(judge.evaluate(t5_records))

    print("\nQwen2.5-3B-Instruct — LLM-as-a-Judge Evaluation\n")
    qwen_judged: Dict[str, Any] = asyncio.run(judge.evaluate(qwen_records))

    print("\nLLaMA-3.2-3B — LLM-as-a-Judge Evaluation\n")
    llama_judged: Dict[str, Any] = asyncio.run(judge.evaluate(llama_records))