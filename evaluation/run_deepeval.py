import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import OpenAIModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
from rag_utils import rag

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
GEN_MODEL = "gpt-4o-mini"
EVAL_MODEL = "gpt-4o"
K = 3
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 512

def build_metrics(eval_model):
    faithfulness = GEval(
        name="Faithfulness",
        criteria="Determine whether the actual output is factually supported by the provided retrieval context.",
        evaluation_steps=[
            "Break the actual output into individual factual claims.",
            "Check whether each claim is supported by the retrieval context.",
            "Penalize claims that are unsupported or contradicted by the retrieval context.",
            "Do not penalize differences in wording when the meaning is supported by the context.",
            "Give a higher score when all important claims are supported by the context.",
        ],
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        threshold=0.85,
        model=eval_model,
    )
    recall = GEval(
        name="Contextual Recall",
        criteria="Determine whether the retrieval context contains the information needed to support the expected output.",
        evaluation_steps=[
            "Break the expected output into its important factual claims.",
            "Check whether each important claim is present or supported by the retrieval context.",
            "Penalize the retrieval context when important information required for the expected output is missing.",
            "Do not penalize differences in wording or presentation.",
            "Give a higher score when the retrieval context contains most or all information needed to produce the expected output.",
        ],
        evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        threshold=0.80,
        model=eval_model,
    )
    precision = GEval(
        name="Contextual Precision",
        criteria="Determine whether the retrieved context is relevant to answering the input question, with relevant information prioritized over irrelevant information.",
        evaluation_steps=[
            "Identify the information in the retrieval context that is relevant to the input question.",
            "Check whether the retrieval context contains unnecessary or unrelated information.",
            "Penalize retrievals that contain a large amount of irrelevant information.",
            "Give more credit when relevant information is prioritized.",
            "Give a higher score when most of the retrieval context is useful for answering the question.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        threshold=0.75,
        model=eval_model,
    )
    relevancy = GEval(
        name="Answer Relevancy",
        criteria="Determine whether the actual output directly addresses the input question and provides an appropriate response without unnecessary information.",
        evaluation_steps=[
            "Check whether the actual output directly answers the input question.",
            "Check whether the response contains information relevant to the question.",
            "Penalize unnecessary, unrelated, or off-topic information.",
            "Do not penalize a concise answer if it fully addresses the question.",
            "Give a higher score when the response is focused and directly useful to the user.",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.75,
        model=eval_model,
    )
    return [faithfulness, recall, precision, relevancy]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["evaluation", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_path = ROOT / "evaluation" / f"{args.dataset}.csv"
    df = pd.read_csv(data_path)
    if args.limit:
        df = df.head(args.limit)

    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE")
    if not api_key or not api_base:
        raise RuntimeError("OPENAI_API_KEY and OPENAI_API_BASE must be set.")

    print(f"Dataset: {args.dataset} | rows: {len(df)}")
    print(f"Generator: {GEN_MODEL} | Judge: {EVAL_MODEL}")
    print(f"k={K}, temperature={TEMPERATURE}, top_p={TOP_P}, max_tokens={MAX_TOKENS}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        str(BACKEND / "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"FAISS vectors: {vectorstore.index.ntotal}")

    eval_model = OpenAIModel(model=EVAL_MODEL, api_key=api_key, base_url=api_base)
    print("Judge connectivity:", eval_model.generate("Reply with exactly: OK"))

    test_cases = []
    for i, row in df.iterrows():
        print(f"Generating RAG answer {len(test_cases)+1}/{len(df)}...")
        answer, retrieved = rag(
            query=row["question"],
            k=K,
            model_name=GEN_MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            vectorstore=vectorstore,
        )
        relevant_context = [c.strip() for c in str(row["relevant_context"]).split("<>") if c.strip()]
        test_cases.append(
            LLMTestCase(
                input=row["question"],
                actual_output=answer,
                expected_output=row["expected_answer"],
                retrieval_context=retrieved,
                context=relevant_context,
            )
        )

    metrics = build_metrics(eval_model)
    results = evaluate(test_cases=test_cases, metrics=metrics)

    rows = []
    for case, result in zip(test_cases, results.test_results):
        scores = {
            m.name.replace(" [GEval]", "").strip().lower(): float(m.score)
            for m in result.metrics_data
        }
        composite = (
            0.35 * scores["faithfulness"]
            + 0.25 * scores["contextual recall"]
            + 0.20 * scores["contextual precision"]
            + 0.20 * scores["answer relevancy"]
        )
        route = "AUTO ANSWER" if composite > 0.75 else ("ASSISTED REVIEW" if composite >= 0.50 else "MANUAL LOOKUP")
        rows.append({
            "Question": case.input,
            "Faithfulness Score": scores["faithfulness"],
            "Contextual Recall Score": scores["contextual recall"],
            "Contextual Precision Score": scores["contextual precision"],
            "Answer Relevancy Score": scores["answer relevancy"],
            "Composite Score": composite,
            "Type": route,
        })

    out = pd.DataFrame(rows)
    out_path = ROOT / "evaluation" / f"{args.dataset}_deepeval_results.csv"
    out.to_csv(out_path, index=False)

    print("\n=== DEEPEVAL RESULTS ===")
    print(out.round(3).to_string(index=False))
    print("\n=== SUMMARY ===")
    print(out[["Faithfulness Score","Contextual Recall Score","Contextual Precision Score","Answer Relevancy Score","Composite Score"]].mean().round(3).to_string())
    print("\nRouting:")
    print(out["Type"].value_counts().to_string())
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
