"""
Patch module: injects environment variables and monkey-patches GptCompletion
to work with local vLLM server instead of OpenAI API.

IMPORTANT: This module must be imported BEFORE any tabqa modules.
"""

import os
import time
import pprint

from local_inference.config import VLLM_BASE_URL, VLLM_API_KEY, MAX_TOKENS, MAX_RETRY

# ===========================================================================
# Step 1: Set environment variables BEFORE importing tabqa modules.
# OpenAI SDK v1.x reads these when api_key/base_url are not explicitly set.
# ===========================================================================
os.environ["OPENAI_API_KEY"] = VLLM_API_KEY
os.environ["OPENAI_BASE_URL"] = VLLM_BASE_URL

# ===========================================================================
# Step 2: Import original modules (env vars now in effect)
# ===========================================================================
from tabqa.GptConnector import prompt2messages  # noqa: E402


# ===========================================================================
# Step 3: Robust response parser for 3B model
# ===========================================================================
def parse_llm_response(original_result):
    """Parse LLM response with robustness for smaller models.

    The original code expects: "SQL: ```code```" or "Answer: ```value```"
    3B models may produce variations like:
    - "sql: SELECT ..." (lowercase)
    - "Answer: some text" (no backticks)
    - Extra whitespace or newlines
    """
    original_result = original_result.strip()

    # Extract answer_type from the first colon-separated token
    answer_type = original_result.split(":")[0].strip()

    # Normalize type names
    type_mapping = {
        "SQL": "SQL", "sql": "SQL", "Sql": "SQL",
        "Python": "Python", "python": "Python", "Py": "Python",
        "Answer": "Answer", "answer": "Answer",
    }
    answer_type = type_mapping.get(answer_type, answer_type)

    # Extract code/answer from backticks
    if "```" in original_result:
        answer = original_result.split("```")[-1]
    else:
        # Fallback: take everything after the first colon
        parts = original_result.split(":", 1)
        answer = parts[1].strip() if len(parts) > 1 else original_result

    return answer_type, answer


# ===========================================================================
# Step 4: Monkey-patched GptCompletion
# ===========================================================================
# Shared client (reuses HTTP connection pool)
from openai import OpenAI  # noqa: E402

_vllm_client = OpenAI()  # reads OPENAI_API_KEY and OPENAI_BASE_URL from env


def patched_gpt_completion(
    engine,
    prompt,
    suffix=None,
    max_tokens=MAX_TOKENS,  # was hardcoded to 128
    temperature=0,
    top_p=1,
    n=1,
    stream=False,
    logprobs=None,
    stop=["```.", "``` "],
    presence_penalty=0,
    frequency_penalty=0,
    best_of=1,
    debug=False,
    prompt_end="\n\n",
    max_retry=MAX_RETRY,  # was 1
):
    """Drop-in replacement for GptCompletion that always uses chat completions
    and points to the local vLLM server via environment variables."""
    messages = prompt2messages(prompt.strip())

    if debug:
        print("=================================== Prompt Messages:")
        pprint.pprint(messages)
        print("===================================")

    last_error = None
    for attempt in range(max_retry):
        try:
            kwargs = {}
            if logprobs is not None:
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = logprobs

            output = _vllm_client.chat.completions.create(
                model=engine,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stream=stream,
                stop=stop,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                **kwargs,
            )

            # Set .text on each choice for backward compatibility
            for choice in output.choices:
                raw_text = choice.message.content or ""
                choice.text = raw_text.strip("`")
                # Rewrite "Answer:" without backticks for consistency
                if "Answer:" in choice.text and "Answer: ```" not in choice.text:
                    choice.text = choice.text.replace("Answer:", "Answer: ```")

            return output

        except Exception as e:
            last_error = e
            print(f"vLLM API error (attempt {attempt + 1}/{max_retry}): {e}")
            if attempt < max_retry - 1:
                time.sleep(2)

    raise ValueError(
        f"vLLM connection failed after {max_retry} retries. Last error: {last_error}"
    )


# ===========================================================================
# Step 5: Apply monkey-patches to tabqa modules
# ===========================================================================
def apply_patches():
    """Apply all monkey-patches. Must be called after importing tabqa modules."""
    import tabqa.GptConnector as connector

    # Replace GptCompletion globally
    connector.GptCompletion = patched_gpt_completion

    # Also patch the name in the module namespace so `from ... import GptCompletion`
    # in other tabqa modules picks up the patched version.
    # Since GptCOTPrompter.py does `from tabqa.GptConnector import *`,
    # it gets GptCompletion at import time. We need to also patch it there.
    try:
        import tabqa.GptCOTPrompter as cot
        cot.GptCompletion = patched_gpt_completion
    except ImportError:
        pass

    try:
        import tabqa.GptCOTPrompter_BeamSeach as beam
        beam.GptCompletion = patched_gpt_completion
    except ImportError:
        pass

    try:
        import tabqa.GptCOTPrompter_SplitFact as split
        split.GptCompletion = patched_gpt_completion
    except ImportError:
        pass

    try:
        import tabqa.GptPAL as pal
        pal.GptCompletion = patched_gpt_completion
    except ImportError:
        pass

    print(f"[patch] GptCompletion patched -> vLLM at {VLLM_BASE_URL}")
    print(f"[patch] parse_llm_response registered")
