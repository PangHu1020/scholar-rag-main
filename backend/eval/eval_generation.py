"""Generation evaluation: dataset-agnostic RAGAS-based functions."""

import time
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config

logger = logging.getLogger(__name__)

DEFAULT_METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), FactualCorrectness()]


def collect_samples(graph, eval_cases: list[dict], verbose: bool = True) -> list[SingleTurnSample]:
    """Run the agent graph on eval_cases and collect RAGAS samples.

    Each case must have "query"; optionally "reference" for FactualCorrectness.
    """
    samples = []
    for i, case in enumerate(eval_cases, 1):
        query = case["query"]
        if verbose:
            print(f"\n[{i}/{len(eval_cases)}] {query}")

        t0 = time.time()
        result = graph.invoke({
            "query": query,
            "messages": [],
            "summary": "",
            "documents": [],
            "sub_queries": [],
            "sub_answers": [],
            "answer": "",
            "citations": [],
        })
        elapsed = time.time() - t0

        answer = result.get("answer", "")
        contexts = [sa["answer"] for sa in result.get("sub_answers", []) if sa.get("answer")]

        if verbose:
            print(f"  {elapsed:.1f}s | sub-queries={len(result.get('sub_answers', []))} | contexts={len(contexts)}")
            print(f"  {answer[:120]}...")

        samples.append(SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts if contexts else [answer],
            reference=case.get("reference"),
        ))

    return samples


def evaluate_generation(
    samples: list[SingleTurnSample],
    llm=None,
    metrics=None,
) -> dict:
    """Run RAGAS evaluation on collected samples.

    Args:
        samples: From collect_samples().
        llm: LangChain LLM for RAGAS evaluation (defaults to Config LLM).
        metrics: RAGAS metric list (defaults to DEFAULT_METRICS).

    Returns:
        Dict of {metric_name: avg_score} plus per-sample DataFrame under "dataframe".
    """
    from langchain_openai import ChatOpenAI

    _llm = llm or ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        api_key=Config.LLM_API_KEY,
        temperature=0,
    )
    _metrics = metrics or DEFAULT_METRICS

    evaluator_llm = LangchainLLMWrapper(_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    )

    dataset = EvaluationDataset(samples=samples)
    result = ragas_evaluate(
        dataset=dataset,
        metrics=_metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
    )

    df = result.to_pandas()
    skip_cols = {"user_input", "response", "retrieved_contexts", "reference", "reference_contexts"}
    metric_cols = [c for c in df.columns if c not in skip_cols]

    summary = {"dataframe": df}
    for col in metric_cols:
        vals = df[col].dropna()
        if len(vals) > 0:
            summary[col] = float(vals.mean())

    return summary
