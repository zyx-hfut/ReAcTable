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
# This sets env vars and monkey-patches GptCompletion.
import local_inference.patch as patch
patch.apply_patches()

# Step 2: Import tabqa modules (env vars already in effect)
from tabqa.GptCOTPrompter_BeamSeach import CodexAnswerCOTExecutor_HighTemperaturMajorityVote  # noqa: E402
from tabqa.GptCOTPrompter import table_formater  # noqa: E402
from tabqa.GptConnector import GptCompletion  # noqa: E402 (this is now the patched version)
from local_inference.config import (  # noqa: E402
    MODEL_NAME, MAX_TOKENS, REPEAT_TIMES, MAX_DEMO, N_THREADS,
    LINE_LIMIT, BASE_PATH, DATA_FILE, TEMPLATE, PROGRAM, RESULTS_DIR,
)
from local_inference.patch import parse_llm_response  # noqa: E402

import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from tqdm import tqdm  # noqa: E402


# ===========================================================================
# Subclass: robust ReAct loop + robust majority voting
# ===========================================================================
class RobustCOTExecutor(CodexAnswerCOTExecutor_HighTemperaturMajorityVote):
    """Subclass that overrides _get_gpt_prediction with robust response
    parsing for 3B models, and _get_gpt_prediction_majority_vote with
    error handling for individual votes."""

    def _get_gpt_prediction(self, maintain_df_ids=False):
        """Override: same ReAct loop logic but with robust parsing and
        consistent response access via .choices[0].text (set by patched
        GptCompletion)."""
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
                engine=self.model,
                prompt=self.prompt,
                max_tokens=MAX_TOKENS,
                temperature=self.temperature,
                top_p=1,
                frequency_penalty=self.frequency_penalty,
                n=1,
                stream=False,
            )

            # Use .text set by patched GptCompletion
            original_result = original_output.choices[0].text.strip('\n')

            # Robust parsing for 3B model
            answer_type, answer = parse_llm_response(original_result)
            self.original_output.append(original_result)

            if iteration_cnt > self.iteration_max_limit:
                # Force answer
                self.prompt += '\nAnswer: ```'
                original_output = GptCompletion(
                    engine=self.model,
                    prompt=self.prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=self.temperature,
                    top_p=1,
                    frequency_penalty=self.frequency_penalty,
                    n=1,
                    stream=False,
                )
                original_result = original_output.choices[0].text.replace('\n', '')
                self.predicted_result = original_result
                break

            elif answer_type == 'Answer':
                # Extract final answer from backtick-delimited content
                if '```' in answer:
                    self.predicted_result = answer.split('```')[-1]
                else:
                    self.predicted_result = answer.strip('`').strip()
                break

            elif answer_type in self.supported_code_types:
                renewed_df = self._executor(self.source_table_df, answer, answer_type)

                # Walk back through historical DataFrames on failure
                i = len(self.series_dfs) - 1
                while i >= 0 and (renewed_df is None):
                    self.source_table_df = self.series_dfs[i]
                    renewed_df = self._executor(self.source_table_df, answer, answer_type)
                    if renewed_df is not None:
                        self.gpt_error = None
                    i -= 1
                self.source_table_df = renewed_df

                if renewed_df is None or answer in self.code_history:
                    # Execution failed or duplicate code -> force answer
                    self.prompt += '\nAnswer: ```'
                    original_output = GptCompletion(
                        engine=self.model,
                        prompt=self.prompt,
                        max_tokens=MAX_TOKENS,
                        temperature=self.temperature,
                        top_p=1,
                        frequency_penalty=self.frequency_penalty,
                        n=1,
                        stream=False,
                    )
                    original_result = original_output.choices[0].text.replace('\n', '')
                    self.predicted_result = original_result
                    break

                self.code_history.append(answer)
                data_table = table_formater(
                    self.source_table_df, permute_df=False, line_limit=self.line_limit
                )
                if not maintain_df_ids:
                    intermediate_prompt_template = \
                        self.prompt_template_dict['intermediate_prompt_template'][answer_type]
                else:
                    intermediate_prompt_template = \
                        self.prompt_template_dict['intermediate_prompt_template'][answer_type] \
                        .replace(':\n', f" (DF{iteration_cnt}):\n")

                self.prompt = (
                    self.prompt.strip('\n')
                    + '\n\n' + original_result + '```.\n\n'
                    + intermediate_prompt_template.format(data_table, self.utterance)
                )
                self.series_dfs.append(renewed_df)

            else:
                # Unsupported code type -> force answer
                self.gpt_error = f'Unsupported code type generated: {answer_type} ({answer})'
                self.prompt += '\nAnswer: ```'
                original_output = GptCompletion(
                    engine=self.model,
                    prompt=self.prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=self.temperature,
                    top_p=1,
                    frequency_penalty=self.frequency_penalty,
                    n=1,
                    stream=False,
                )
                original_result = original_output.choices[0].text.replace('\n', '')
                self.predicted_result = original_result
                break

        self.prompt = self.prompts[-1]

    def _get_gpt_prediction_majority_vote(
        self, NNDemo=False, ft=None, repeat_times=REPEAT_TIMES, maintain_df_ids=False
    ):
        """Override: adds try/except around each vote to prevent crashes."""
        all_predictions = []
        for vote_idx in range(repeat_times):
            try:
                self._read_data()
                self._gen_gpt_prompt(NNDemo, ft, maintain_df_ids=maintain_df_ids)
                self._get_gpt_prediction(maintain_df_ids=maintain_df_ids)
                if self.predicted_result is not None and self.predicted_result != "":
                    all_predictions.append(self.predicted_result)
                else:
                    all_predictions.append("")
            except Exception as e:
                print(f"  [Vote {vote_idx + 1}/{repeat_times}] Error: {e}")
                traceback.print_exc()
                all_predictions.append("")

        self.all_predictions = all_predictions
        from collections import Counter
        valid_predictions = [p for p in all_predictions if p]
        if valid_predictions:
            majority = Counter(valid_predictions).most_common(1)[0][0]
        else:
            majority = ""
        self.predicted_result = majority


# ===========================================================================
# Worker function for parallel execution
# ===========================================================================
def process_single_question(i, dataset, max_demo, model, template, program, base_path):
    """Process a single question from the dataset."""
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
            log = executor._log_dict()
            return log
        except Exception as e:
            last_error = str(e)
            log = {
                "id": dataset.iloc[i]["id"],
                "uncaught_err": last_error,
            }
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

    # Load dataset
    dataset = pd.read_csv(DATA_FILE, sep="\t")
    total = min(args.limit, dataset.shape[0])
    print(f"Dataset: {dataset.shape[0]} questions, running {total}")
    print(f"Model: {MODEL_NAME}, demos: {args.demo}, votes: {args.repeat}, threads: {args.threads}")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_file = os.path.join(
        RESULTS_DIR,
        f"RobustCOTExecutor_{TEMPLATE}_{PROGRAM}"
        f"_limit{total}_model{MODEL_NAME}"
        f"_votes{args.repeat}_demo{args.demo}.json",
    )

    # Run evaluation
    logs = Parallel(n_jobs=args.threads, require="sharedmem")(
        delayed(process_single_question)(
            i, dataset, args.demo, MODEL_NAME, TEMPLATE, PROGRAM, BASE_PATH
        )
        for i in tqdm(range(total))
    )

    # Save results
    json.dump(logs, open(output_file, "w"), indent=4)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    errors = sum(1 for log in logs if "uncaught_err" in log)
    print(f"Completed: {total - errors}/{total} ({errors} errors)")


if __name__ == "__main__":
    main()
