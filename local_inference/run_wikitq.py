"""
Main script for running ReAcTable on WikiTQ with local Qwen2.5-3B-Instruct via vLLM.

Usage:
    1. Start vLLM server first (see setup_env.sh)
    2. conda activate reactable-qwen
    3. python local_inference/run_wikitq.py --limit 5
"""

import argparse
import json
import os
import sys
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Step 1: Apply patches BEFORE importing tabqa experiment modules.
import local_inference.patch as patch
patch.apply_patches()

# Step 2: Import tabqa modules
from tabqa.GptCOTPrompter_BeamSeach import CodexAnswerCOTExecutor_HighTemperaturMajorityVote
from tabqa.GptCOTPrompter import table_formater
from tabqa.GptConnector import GptCompletion
from local_inference.config import (
    MODEL_NAME, MAX_TOKENS, REPEAT_TIMES, MAX_DEMO, N_THREADS,
    LINE_LIMIT, BASE_PATH, DATA_FILE, TEMPLATE, PROGRAM, RESULTS_DIR,
)
from local_inference.patch import parse_llm_response

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def _clean_answer(text):
    """Clean a predicted answer by removing format artifacts.

    Handles cases where the model outputs SQL/Python instead of Answer,
    or includes backticks/formatting in the answer.
    """
    if not text:
        return ""
    text = text.strip()
    # Remove outer backticks
    text = text.strip("`")
    text = text.strip()
    # Remove trailing period
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def _extract_answer_from_response(original_result):
    """Try to extract a clean answer from any LLM response.

    Works for: "Answer: ```value```", "Answer: value", "value"
    Returns a clean answer string, or empty string if not an Answer.
    """
    result_type, answer = parse_llm_response(original_result)
    if result_type == "Answer":
        return _clean_answer(answer)
    return ""


def _force_answer(GptCompletion_fn, engine, prompt, temperature, frequency_penalty, max_tokens):
    """Call LLM with 'Answer: ```' suffix, parse response robustly.

    Returns a clean answer string. Retries once with explicit instruction
    if the model outputs SQL/Python instead of Answer.
    """
    # First attempt: standard force-answer
    forced_prompt = prompt.rstrip("\n") + "\n\nAnswer: ```"
    output = GptCompletion_fn(
        engine=engine, prompt=forced_prompt,
        max_tokens=max_tokens, temperature=temperature,
        top_p=1, frequency_penalty=frequency_penalty,
        n=1, stream=False,
    )
    raw = output.choices[0].text.replace("\n", " ").strip()
    answer = _extract_answer_from_response(raw)
    if answer:
        return answer

    # Second attempt: explicit instruction
    retry_prompt = prompt.rstrip("\n") + (
        "\n\nYou must now answer directly. "
        "Output ONLY the answer value wrapped in backticks, like: Answer: ```your answer```\n"
        "Answer: ```"
    )
    try:
        output = GptCompletion_fn(
            engine=engine, prompt=retry_prompt,
            max_tokens=max_tokens, temperature=0,  # greedy for cleaner output
            top_p=1, frequency_penalty=frequency_penalty,
            n=1, stream=False,
        )
        raw = output.choices[0].text.replace("\n", " ").strip()
        answer = _extract_answer_from_response(raw)
        if answer:
            return answer
    except Exception:
        pass

    # Last resort: return raw text with format artifacts removed
    return _clean_answer(raw)


# ===========================================================================
# Subclass: robust ReAct loop + robust majority voting
# ===========================================================================
class RobustCOTExecutor(CodexAnswerCOTExecutor_HighTemperaturMajorityVote):

    def _get_gpt_prediction(self, maintain_df_ids=False):
        self.prompts = []
        self.source_table_df.columns = \
            [c.replace('\n', ' ').replace(' ', '_').lower()
             for c in self.source_table_df.columns.tolist()]
        self.code_history = []
        iteration_cnt = 0

        while True:
            iteration_cnt += 1
            self.prompts.append(self.prompt)

            original_output = GptCompletion(
                engine=self.model, prompt=self.prompt,
                max_tokens=MAX_TOKENS, temperature=self.temperature,
                top_p=1, frequency_penalty=self.frequency_penalty,
                n=1, stream=False,
            )
            original_result = original_output.choices[0].text.strip("\n")
            answer_type, answer = parse_llm_response(original_result)
            self.original_output.append(original_result)

            if iteration_cnt > self.iteration_max_limit:
                self.predicted_result = _force_answer(
                    GptCompletion, self.model, self.prompt,
                    self.temperature, self.frequency_penalty, MAX_TOKENS,
                )
                break

            elif answer_type == "Answer":
                self.predicted_result = _clean_answer(answer)
                break

            elif answer_type in self.supported_code_types:
                renewed_df = self._executor(self.source_table_df, answer, answer_type)

                i = len(self.series_dfs) - 1
                while i >= 0 and (renewed_df is None):
                    self.source_table_df = self.series_dfs[i]
                    renewed_df = self._executor(self.source_table_df, answer, answer_type)
                    if renewed_df is not None:
                        self.gpt_error = None
                    i -= 1
                self.source_table_df = renewed_df

                if renewed_df is None or answer in self.code_history:
                    self.predicted_result = _force_answer(
                        GptCompletion, self.model, self.prompt,
                        self.temperature, self.frequency_penalty, MAX_TOKENS,
                    )
                    break

                self.code_history.append(answer)
                data_table = table_formater(
                    self.source_table_df, permute_df=False, line_limit=self.line_limit,
                )
                if not maintain_df_ids:
                    tmpl = self.prompt_template_dict["intermediate_prompt_template"][answer_type]
                else:
                    tmpl = self.prompt_template_dict["intermediate_prompt_template"][answer_type] \
                        .replace(":\n", f" (DF{iteration_cnt}):\n")

                self.prompt = (
                    self.prompt.strip("\n")
                    + "\n\n" + original_result + "```.\n\n"
                    + tmpl.format(data_table, self.utterance)
                )
                self.series_dfs.append(renewed_df)

            else:
                self.gpt_error = f"Unsupported code type: {answer_type} ({answer})"
                self.predicted_result = _force_answer(
                    GptCompletion, self.model, self.prompt,
                    self.temperature, self.frequency_penalty, MAX_TOKENS,
                )
                break

        self.prompt = self.prompts[-1]

    def _get_gpt_prediction_majority_vote(
        self, NNDemo=False, ft=None, repeat_times=REPEAT_TIMES, maintain_df_ids=False
    ):
        all_predictions = []
        for vote_idx in range(repeat_times):
            try:
                self._read_data()
                self._gen_gpt_prompt(NNDemo, ft, maintain_df_ids=maintain_df_ids)
                self._get_gpt_prediction(maintain_df_ids=maintain_df_ids)
                if self.predicted_result:
                    all_predictions.append(self.predicted_result)
                else:
                    all_predictions.append("")
            except Exception as e:
                print(f"  [Vote {vote_idx + 1}/{repeat_times}] Error: {e}")
                traceback.print_exc()
                all_predictions.append("")

        self.all_predictions = all_predictions
        from collections import Counter
        valid = [p for p in all_predictions if p]
        if valid:
            self.predicted_result = Counter(valid).most_common(1)[0][0]
        else:
            self.predicted_result = ""


# ===========================================================================
# Worker function for parallel execution
# ===========================================================================
def process_single_question(i, dataset, max_demo, model, template, program, base_path):
    max_retry = 3
    last_error = None
    while max_retry > 0:
        try:
            executor = RobustCOTExecutor(
                prompt_template_json=f"prompt_template/{template}.json",
                qid=dataset.iloc[i]["id"],
                utterance=dataset.iloc[i]["utterance"],
                source_csv=dataset.iloc[i]["context"],
                target_value=dataset.iloc[i]["targetValue"],
                base_path=base_path,
                demo_file=f"few-shot-demo/WikiTQ-{program}.json",
            )
            executor.max_demo = max_demo
            executor.model = model
            executor._gen_gpt_prompt(False)
            executor._get_gpt_prediction_majority_vote(repeat_times=REPEAT_TIMES)
            return executor._log_dict()
        except Exception as e:
            last_error = str(e)
            log = {"id": dataset.iloc[i]["id"], "uncaught_err": last_error}
            if "context length" in last_error or "maximum context" in last_error:
                return log
            max_retry -= 1
    return log


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run ReAcTable on WikiTQ with local Qwen model")
    parser.add_argument("--limit", type=int, default=5, help="Max number of questions to process")
    parser.add_argument("--repeat", type=int, default=REPEAT_TIMES, help="Majority vote repetitions")
    parser.add_argument("--threads", type=int, default=N_THREADS, help="Number of parallel threads")
    parser.add_argument("--demo", type=int, default=MAX_DEMO, help="Number of few-shot demos")
    args = parser.parse_args()

    dataset = pd.read_csv(DATA_FILE, sep="\t")
    total = min(args.limit, dataset.shape[0])
    print(f"Dataset: {dataset.shape[0]} questions, running {total}")
    print(f"Model: {MODEL_NAME}, demos: {args.demo}, votes: {args.repeat}, threads: {args.threads}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_file = os.path.join(
        RESULTS_DIR,
        f"RobustCOTExecutor_{TEMPLATE}_{PROGRAM}"
        f"_limit{total}_model{MODEL_NAME}"
        f"_votes{args.repeat}_demo{args.demo}.json",
    )

    logs = Parallel(n_jobs=args.threads, require="sharedmem")(
        delayed(process_single_question)(
            i, dataset, args.demo, MODEL_NAME, TEMPLATE, PROGRAM, BASE_PATH
        )
        for i in tqdm(range(total))
    )

    json.dump(logs, open(output_file, "w"), indent=4)
    print(f"\nResults saved to: {output_file}")

    errors = sum(1 for log in logs if "uncaught_err" in log)
    print(f"Completed: {total - errors}/{total} ({errors} errors)")


if __name__ == "__main__":
    main()
