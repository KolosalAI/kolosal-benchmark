"""
Concurrent request tester CLI (single file)

Inputs:
- --base-url (str)
- --api-key (str, optional, default "api-key")
- --model (str)
- --length (str, optional, default short)
- --request / --requests (int, optional, default 1)
- --concurrency (int, optional, default 1)
- --max-tokens (int, optional, default 128)
- --temperature (float, optional, default 0)
- --capture-responses (yes/no, optional, default no)
- --output-file (str, responses jsonl path if capturing, default responses.json)
- --summary-file (str, optional path to save summary metrics JSON)

Outputs:
- Prints param table
- After test: shows success count, total time, RPS, tokens/sec (if available),
  avg latency (TTFB), p50 latency, p95 latency, token I/O totals,
  tokens/sec per request (avg), tokens/sec throughput and per worker.
- Optionally writes responses to output file, and metrics to summary file.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def yes_no(s: str) -> bool:
    return str(s).strip().lower() in {"y", "yes", "true", "1", "on"}


def mask(s: Optional[str], visible: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= visible:
        return "*" * len(s)
    return ("*" * (len(s) - visible)) + s[-visible:]


def get_token_encoder(model_name: str):
    """Get the appropriate tiktoken encoder for the model"""
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        # Try to get the encoding for the specific model
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # If model not found, use a reasonable default
        try:
            # For most OpenAI-compatible models, cl100k_base is a good default
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def calculate_input_tokens(messages: List[Dict[str, str]], model_name: str) -> int:
    """Calculate input tokens from messages using tiktoken"""
    if not TIKTOKEN_AVAILABLE:
        return 0
    
    encoder = get_token_encoder(model_name)
    if not encoder:
        return 0
    
    try:
        # Calculate tokens for the messages
        # This follows the OpenAI token counting methodology
        total_tokens = 0
        
        # Add tokens for each message
        for message in messages:
            # Each message has some overhead tokens
            total_tokens += 4  # Message overhead
            
            for key, value in message.items():
                if isinstance(value, str):
                    total_tokens += len(encoder.encode(value))
                    if key == "name":  # Name field has additional overhead
                        total_tokens += 1
        
        # Add conversation overhead
        total_tokens += 2  # Conversation overhead
        
        return total_tokens
    except Exception:
        return 0


def calculate_output_tokens(response_content: str, model_name: str) -> int:
    """Calculate output tokens from response content using tiktoken"""
    if not TIKTOKEN_AVAILABLE or not response_content:
        return 0
    
    encoder = get_token_encoder(model_name)
    if not encoder:
        return 0
    
    try:
        return len(encoder.encode(response_content))
    except Exception:
        return 0


def now() -> float:
    return time.perf_counter()


def extract_token_usage(obj: Any) -> Tuple[Optional[int], Optional[int]]:
    """Try to extract (input_tokens, output_tokens) from common response shapes."""
    try:
        if not isinstance(obj, dict):
            return (None, None)

        candidates: List[Dict[str, Any]] = []
        for key in ("usage", "token_usage", "metrics"):
            if key in obj and isinstance(obj[key], dict):
                candidates.append(obj[key])
        candidates.append(obj)

        def get_pair(d: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
            if "prompt_tokens" in d or "completion_tokens" in d:
                return (
                    int(d.get("prompt_tokens") or 0) if d.get("prompt_tokens") is not None else None,
                    int(d.get("completion_tokens") or 0) if d.get("completion_tokens") is not None else None,
                )
            if "input_tokens" in d or "output_tokens" in d:
                return (
                    int(d.get("input_tokens") or 0) if d.get("input_tokens") is not None else None,
                    int(d.get("output_tokens") or 0) if d.get("output_tokens") is not None else None,
                )
            t = d.get("tokens") if isinstance(d.get("tokens"), dict) else None
            if t:
                inp = t.get("input") if t.get("input") is not None else t.get("in")
                out = t.get("output") if t.get("output") is not None else t.get("out")
                if inp is not None or out is not None:
                    return (int(inp) if inp is not None else None, int(out) if out is not None else None)
            return (None, None)

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            i, o = get_pair(cand)
            if i is not None or o is not None:
                return (i, o)
    except Exception:
        pass
    return (None, None)


@dataclass
class SingleResult:
    request_id: int
    ok: bool
    status_code: Optional[int]
    error: Optional[str]
    latency_s: float  # time to first byte/chunk
    ttft_s: Optional[float]  # time to first token (for streaming)
    duration_s: float  # full response time
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    response_json: Optional[Any]
    streaming_chunks: Optional[int]  # number of chunks received


def do_post(
    idx: int,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    timeout_s: float = 60.0,
) -> SingleResult:
    start = now()
    
    # Calculate input tokens from the request body
    model_name = body.get("model", "gpt-3.5-turbo")  # Default fallback
    messages = body.get("messages", [])
    input_tokens = calculate_input_tokens(messages, model_name)
    
    # Add streaming to the request body
    body_with_streaming = body.copy()
    body_with_streaming["stream"] = True
    
    try:
        data = json.dumps(body_with_streaming).encode("utf-8")
        req = urllib_request.Request(url=url, data=data, headers=headers, method="POST")
        resp = urllib_request.urlopen(req, timeout=timeout_s)
        
        status = getattr(resp, "status", None) or getattr(resp, "code", None)
        if not (status and 200 <= int(status) < 300):
            # Non-successful status, read error and return
            try:
                error_data = resp.read()
                error_text = error_data.decode("utf-8", errors="replace") if error_data else f"HTTP {status}"
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
            latency = now() - start
            return SingleResult(idx, False, int(status) if status else None, error_text, latency, None, latency, input_tokens, 0, None, 0)
        
        # Process streaming response
        first_chunk_time = None
        ttft = None
        chunks_received = 0
        accumulated_content = ""
        complete_response_data = []
        
        try:
            # Read the streaming response line by line
            for line in resp:
                chunk_time = now()
                if first_chunk_time is None:
                    first_chunk_time = chunk_time
                    latency = chunk_time - start  # TTFB (time to first chunk)
                
                # Decode the line
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue
                    
                # Skip SSE prefixes
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                
                # Skip ping/heartbeat lines
                if line_str in ["", "[DONE]"]:
                    continue
                
                try:
                    # Parse JSON chunk
                    chunk_data = json.loads(line_str)
                    complete_response_data.append(chunk_data)
                    chunks_received += 1
                    
                    # Extract content from chunk for TTFT measurement
                    if ttft is None and chunks_received > 0:
                        # Look for first actual content token
                        choices = chunk_data.get("choices", [])
                        if choices and isinstance(choices, list):
                            delta = choices[0].get("delta", {})
                            if isinstance(delta, dict) and delta.get("content"):
                                ttft = chunk_time - start
                    
                    # Accumulate content for token calculation
                    choices = chunk_data.get("choices", [])
                    if choices and isinstance(choices, list):
                        delta = choices[0].get("delta", {})
                        if isinstance(delta, dict) and "content" in delta:
                            content = delta.get("content", "")
                            if content:
                                accumulated_content += content
                                
                except json.JSONDecodeError:
                    # Skip invalid JSON lines
                    continue
                    
        finally:
            try:
                resp.close()
            except Exception:
                pass
        
        duration = now() - start
        
        # Calculate output tokens from accumulated content
        output_tokens = calculate_output_tokens(accumulated_content, model_name)
        
        # Create a combined response object
        combined_response = {
            "choices": [{"message": {"role": "assistant", "content": accumulated_content}}],
            "streaming_chunks": complete_response_data,
            "total_chunks": chunks_received
        }
        
        return SingleResult(
            request_id=idx,
            ok=True,
            status_code=int(status) if status else None,
            error=None,
            latency_s=latency if first_chunk_time else (now() - start),
            ttft_s=ttft,
            duration_s=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_json=combined_response,
            streaming_chunks=chunks_received,
        )
    except HTTPError as he:
        latency = now() - start
        try:
            body_content = he.read()
            btxt = body_content.decode("utf-8", errors="replace") if body_content else str(he)
        except Exception:
            btxt = str(he)
        return SingleResult(idx, False, he.code, btxt, latency, None, latency, input_tokens, 0, None, 0)
    except URLError as ue:
        latency = now() - start
        msg = str(ue.reason) if hasattr(ue, "reason") else str(ue)
        return SingleResult(idx, False, None, msg, latency, None, latency, input_tokens, 0, None, 0)
    except Exception as e:
        latency = now() - start
        return SingleResult(idx, False, None, f"{e}\n{traceback.format_exc()}", latency, None, latency, input_tokens, 0, None, 0)


def print_params_table(params: List[Tuple[str, str]]) -> None:
    key_w = max(len(k) for k, _ in params)
    val_w = max(len(str(v)) for _, v in params)
    total_w = key_w + 3 + val_w
    line = "+" + ("-" * (total_w + 2)) + "+"
    print(line)
    for k, v in params:
        print(f"| {k.ljust(key_w)} : {str(v).ljust(val_w)} |")
    print(line)


def print_results_table(results: List[Tuple[str, str]]) -> None:
    """Print results in a nicely formatted table"""
    if not results:
        return
    
    # Calculate column widths
    metric_w = max(len(metric) for metric, _ in results)
    value_w = max(len(str(value)) for _, value in results)
    total_w = metric_w + 3 + value_w
    
    # Create table borders
    line = "+" + ("-" * (total_w + 2)) + "+"
    
    print("\nTest Results:")
    print(line)
    for metric, value in results:
        print(f"| {metric.ljust(metric_w)} : {str(value).ljust(value_w)} |")
    print(line)


def print_error_table(errors: List[Tuple[str, str, str]]) -> None:
    """Print errors in a nicely formatted table"""
    if not errors:
        return
        
    # Calculate column widths
    id_w = max(len(str(req_id)) for req_id, _, _ in errors)
    status_w = max(len(str(status)) for _, status, _ in errors)
    error_w = max(len(str(error)[:50]) for _, _, error in errors)  # Limit error width
    
    # Ensure minimum widths
    id_w = max(id_w, len("Request ID"))
    status_w = max(status_w, len("Status"))
    error_w = max(error_w, len("Error"))
    
    total_w = id_w + status_w + error_w + 6  # 6 for separators
    line = "+" + "-" * (id_w + 2) + "+" + "-" * (status_w + 2) + "+" + "-" * (error_w + 2) + "+"
    
    print("\nFailed Requests (First 3):")
    print(line)
    print(f"| {'Request ID'.ljust(id_w)} | {'Status'.ljust(status_w)} | {'Error'.ljust(error_w)} |")
    print(line)
    
    for req_id, status, error in errors:
        truncated_error = (error[:47] + "...") if len(error) > 50 else error
        print(f"| {str(req_id).ljust(id_w)} | {str(status).ljust(status_w)} | {truncated_error.ljust(error_w)} |")
    
    print(line)


def extract_response_content(response_json: Any) -> str:
    """Extract the actual response content from the API response"""
    try:
        if not isinstance(response_json, dict):
            return str(response_json)[:200] if response_json else ""
        
        # Common OpenAI API response structure
        choices = response_json.get("choices", [])
        if choices and isinstance(choices, list) and len(choices) > 0:
            choice = choices[0]
            if isinstance(choice, dict):
                # Check for message content
                message = choice.get("message", {})
                if isinstance(message, dict) and "content" in message:
                    return str(message["content"])
                # Check for text content (older format)
                if "text" in choice:
                    return str(choice["text"])
        
        # If no standard format found, return first reasonable text content
        for key in ["content", "text", "message", "output"]:
            if key in response_json and isinstance(response_json[key], str):
                return response_json[key]
                
        # Last resort - return string representation
        return str(response_json)[:200]
    except Exception:
        return str(response_json)[:200] if response_json else ""


def print_last_response(results: List[SingleResult]) -> None:
    """Print the last successful response content (truncated)"""
    if not results:
        return
        
    # Find the last successful response
    successful_results = [r for r in results if r.ok and r.response_json]
    if not successful_results:
        return
    
    # Sort by request_id and get the last one
    last_result = sorted(successful_results, key=lambda x: x.request_id)[-1]
    content = extract_response_content(last_result.response_json)
    
    if content:
        # Truncate content for display
        max_length = 300
        truncated_content = content[:max_length] + "..." if len(content) > max_length else content
        
        print(f"\nLast Response (Request #{last_result.request_id}):")
        print("+" + "-" * 82 + "+")
        
        # Split content into lines and wrap
        lines = truncated_content.split('\n')
        for line in lines[:10]:  # Show max 10 lines
            # Wrap long lines
            while len(line) > 80:
                print(f"| {line[:80].ljust(80)} |")
                line = line[80:]
            if line:
                print(f"| {line.ljust(80)} |")
        
        if len(lines) > 10:
            print(f"| {'... (truncated)'.ljust(80)} |")
            
        print("+" + "-" * 82 + "+")


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    import math as _m
    k = max(1, min(n, int(_m.ceil((p / 100.0) * n)))) - 1
    return s[k]

def _resolve_prompt_file() -> Optional[str]:
    candidates = [
        os.path.join(os.getcwd(), "prompt.json"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.json"),
    ]
    for pth in candidates:
        if os.path.exists(pth):
            return pth
    return None


def load_prompt_by_length(length: str) -> str:
    path = _resolve_prompt_file()
    if not path:
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            sel = str(length).strip().lower()
            # find exact match first
            for item in data:
                if isinstance(item, dict) and str(item.get("length", "")).strip().lower() == sel:
                    return str(item.get("prompt", ""))
            # fallback: first element's prompt
            for item in data:
                if isinstance(item, dict) and "prompt" in item:
                    return str(item.get("prompt", ""))
        elif isinstance(data, dict):
            # support alt shape: {"short":"...","medium":"...","long":"..."}
            val = data.get(length) or data.get(length.lower())
            if isinstance(val, str):
                return val
    except Exception:
        pass
    return ""


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Concurrent request tester (single-file).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", required=True, help="Base URL (e.g., https://api.openai.com)")
    p.add_argument("--api-key", default="api-key", help="API key for Authorization header (optional)")
    p.add_argument("--model", required=True, help="Model identifier for the request body")
    p.add_argument("--length", default="short", help="Length hint (short/medium/long)")
    p.add_argument("--request", "--requests", dest="requests", type=int, default=1, help="Total number of requests to send")
    p.add_argument("--concurrency", type=int, default=1, help="Number of concurrent workers")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=128, help="Max tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--capture-responses", default="no", help="Capture responses to output file (yes/no)")
    p.add_argument("--output-file", default="responses.json", help="Output file for captured responses (JSON Lines)")
    p.add_argument("--summary-file", default=None, help="Optional path to save summary metrics as JSON")
    p.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")

    args = p.parse_args(argv)

    # Construct the full API URL
    base_url = args.base_url.rstrip('/')  # Remove trailing slash if present
    if not base_url.endswith('/v1/chat/completions'):
        api_url = f"{base_url}/v1/chat/completions"
    else:
        api_url = base_url

    capture = yes_no(args.capture_responses)
    params_table = [
        ("base_url", args.base_url),
        ("api_key", mask(args.api_key)),
        ("model", args.model),
        ("length", args.length),
        ("requests", str(args.requests)),
        ("concurrency", str(args.concurrency)),
        ("max_tokens", str(args.max_tokens)),
        ("temperature", str(args.temperature)),
        ("capture_responses", "yes" if capture else "no"),
        ("output_file", args.output_file if capture else "-"),
        ("summary_file", args.summary_file if args.summary_file else "-"),
        ("timeout", f"{args.timeout}s"),
    ]

    print("Parameters:")
    print_params_table(params_table)

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # Prepare request bodies: OpenAI-compatible chat completion format
    prompt_text = load_prompt_by_length(args.length)
    bodies: List[Dict[str, Any]] = []
    for i in range(args.requests):
        bodies.append(
            {
                "model": args.model,
                "messages": [
                    {"role": "user", "content": prompt_text}
                ],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            }
        )

    results: List[SingleResult] = []
    wall_start = now()
    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = [
            ex.submit(do_post, i + 1, api_url, headers, bodies[i], args.timeout)
            for i in range(args.requests)
        ]
        for fut in cf.as_completed(futures):
            results.append(fut.result())
    wall_end = now()

    if capture:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                for r in sorted(results, key=lambda x: x.request_id):
                    # Get the corresponding request body for full conversation context
                    request_body = bodies[r.request_id - 1] if r.request_id <= len(bodies) else None
                    out = {
                        "request_id": r.request_id,
                        "ok": r.ok,
                        "status_code": r.status_code,
                        "latency_ms": round(r.latency_s * 1000, 3),
                        "ttft_ms": round(r.ttft_s * 1000, 3) if r.ttft_s is not None else None,
                        "duration_ms": round(r.duration_s * 1000, 3),
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "streaming_chunks": r.streaming_chunks,
                        "request": request_body,  # Full conversation context
                        "response": r.response_json,
                        "error": r.error,
                    }
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"Responses captured to: {args.output_file}")
        except Exception as e:
            print(f"Failed to write responses: {e}")

    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]
    total = len(results)
    ok_count = len(ok_results)
    fail_count = len(fail_results)

    latencies = [r.latency_s for r in ok_results]
    avg_latency_ms = (sum(latencies) / ok_count * 1000) if ok_count else float("nan")
    p50_latency_ms = percentile(latencies, 50) * 1000 if ok_count else float("nan")
    p95_latency_ms = percentile(latencies, 95) * 1000 if ok_count else float("nan")

    # TTFT (Time to First Token) metrics
    ttfts = [r.ttft_s for r in ok_results if r.ttft_s is not None]
    avg_ttft_ms = (sum(ttfts) / len(ttfts) * 1000) if ttfts else float("nan")
    p50_ttft_ms = percentile(ttfts, 50) * 1000 if ttfts else float("nan")
    p95_ttft_ms = percentile(ttfts, 95) * 1000 if ttfts else float("nan")

    durations = [r.duration_s for r in ok_results]
    avg_duration_ms = (sum(durations) / ok_count * 1000) if ok_count else float("nan")

    # Streaming metrics
    total_chunks = sum((r.streaming_chunks or 0) for r in ok_results)
    avg_chunks_per_request = (total_chunks / ok_count) if ok_count else 0

    # Include input tokens from all requests (both successful and failed)
    # since we calculate input tokens locally regardless of API response
    input_tokens = sum((r.input_tokens or 0) for r in results)
    # Only include output tokens from successful requests
    output_tokens = sum((r.output_tokens or 0) for r in ok_results)

    # Per-request tokens/sec (avg over requests with known tokens)
    per_req_tps: List[float] = []
    for r in ok_results:
        out = r.output_tokens if r.output_tokens is not None else 0
        if out and r.duration_s > 0:
            per_req_tps.append(out / r.duration_s)
    avg_per_req_tps = sum(per_req_tps) / len(per_req_tps) if per_req_tps else float("nan")

    wall_elapsed = wall_end - wall_start
    rps = (total / wall_elapsed) if wall_elapsed > 0 else float("nan")
    ok_rps = (ok_count / wall_elapsed) if wall_elapsed > 0 else float("nan")
    input_tps = (input_tokens / wall_elapsed) if wall_elapsed > 0 else float("nan")
    output_tps = (output_tokens / wall_elapsed) if wall_elapsed > 0 else float("nan")
    throughput_per_worker = (output_tps / args.concurrency) if args.concurrency > 0 and not math.isnan(output_tps) else float("nan")

    # Prepare results for table display
    results_data = [
        ("Successful requests", str(ok_count)),
        ("Total requests", str(total)),
        ("Failures", str(fail_count)),
        ("Total execution time", f"{wall_elapsed:.3f} s"),
        ("Requests per second", f"{rps:.2f}" if not math.isnan(rps) else "N/A"),
        ("Avg latency (TTFB)", f"{avg_latency_ms:.1f} ms" if not math.isnan(avg_latency_ms) else "N/A"),
        ("p50 latency (TTFB)", f"{p50_latency_ms:.1f} ms" if not math.isnan(p50_latency_ms) else "N/A"),
        ("p95 latency (TTFB)", f"{p95_latency_ms:.1f} ms" if not math.isnan(p95_latency_ms) else "N/A"),
        ("Avg TTFT", f"{avg_ttft_ms:.1f} ms" if not math.isnan(avg_ttft_ms) else "N/A"),
        ("p50 TTFT", f"{p50_ttft_ms:.1f} ms" if not math.isnan(p50_ttft_ms) else "N/A"),
        ("p95 TTFT", f"{p95_ttft_ms:.1f} ms" if not math.isnan(p95_ttft_ms) else "N/A"),
        ("Avg duration (RTT)", f"{avg_duration_ms:.1f} ms" if not math.isnan(avg_duration_ms) else "N/A"),
        ("Total input tokens", str(input_tokens)),
        ("Total output tokens", str(output_tokens)),
        ("Total streaming chunks", str(total_chunks)),
        ("Avg chunks per request", f"{avg_chunks_per_request:.1f}"),
        ("Tokens/sec (input)", f"{input_tps:.2f}" if not math.isnan(input_tps) else "N/A"),
        ("Tokens/sec (output)", f"{output_tps:.2f}" if not math.isnan(output_tps) else "N/A"),
        ("Tokens/sec per request", f"{avg_per_req_tps:.2f}" if not math.isnan(avg_per_req_tps) else "N/A"),
        ("Throughput tps/worker", f"{throughput_per_worker:.2f}" if not math.isnan(throughput_per_worker) else "N/A"),
    ]

    print_results_table(results_data)

    # Display the last response
    print_last_response(ok_results)

    if fail_results:
        error_data = []
        for r in fail_results[:3]:
            error_data.append((
                f"#{r.request_id}",
                str(r.status_code) if r.status_code else "N/A",
                str(r.error) if r.error else "Unknown error"
            ))
        print_error_table(error_data)

    summary = {
        "successful_requests": ok_count,
        "total_requests": total,
        "failures": fail_count,
        "total_time_s": wall_elapsed,
        "requests_per_second": rps,
        "successful_rps": ok_rps,
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "avg_ttft_ms": avg_ttft_ms,
        "p50_ttft_ms": p50_ttft_ms,
        "p95_ttft_ms": p95_ttft_ms,
        "avg_duration_ms": avg_duration_ms,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_streaming_chunks": total_chunks,
        "avg_chunks_per_request": avg_chunks_per_request,
        "input_tokens_per_second": input_tps,
        "output_tokens_per_second": output_tps,
        "tokens_per_second_per_request": avg_per_req_tps,
        "throughput_tokens_per_second_per_worker": throughput_per_worker,
        "concurrency": args.concurrency,
    }

    if args.summary_file:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.summary_file)), exist_ok=True)
            with open(args.summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"Summary saved to: {args.summary_file}")
        except Exception as e:
            print(f"Failed to write summary: {e}")

    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
