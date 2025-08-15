# Kolosal Benchmark

A comprehensive concurrent request tester designed for benchmarking OpenAI-compatible API endpoints. This tool helps you measure server performance, latency, and throughput under various load conditions.

## Features

- **Concurrent Load Testing**: Send multiple requests simultaneously with configurable concurrency
- **OpenAI-Compatible**: Works with any OpenAI-compatible chat completion API
- **Local Token Calculation**: Accurate token counting using tiktoken, independent of API responses
- **Detailed Metrics**: Comprehensive performance analysis including latency percentiles, tokens/second, and throughput
- **Flexible Prompts**: Support for different prompt lengths (short, medium, long)
- **Response Capture**: Optional saving of all responses for detailed analysis
- **Summary Export**: Export metrics to JSON for further processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KolosalAI/kolosal-benchmark.git
cd kolosal-benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Prompt Configuration

Edit `prompt.json` to configure your test prompts:

```json
[
    {
        "length": "short",
        "context_size": 0,
        "prompt": "Write a hello world program in Python."
    },
    {
        "length": "medium",
        "context_size": 0,
        "prompt": "Explain the concept of machine learning and provide a simple example implementation."
    },
    {
        "length": "long",
        "context_size": 0,
        "prompt": "Write a comprehensive tutorial on building a REST API with authentication, including code examples, best practices, and deployment considerations."
    }
]
```

## Usage

### Basic Usage

```bash
python benchmark.py --base-url https://api.openai.com --api-key your-api-key --model gpt-3.5-turbo
```

### Advanced Usage

```bash
python benchmark.py \
  --base-url https://your-server.com \
  --api-key your-api-key \
  --model your-model-name \
  --length medium \
  --requests 100 \
  --concurrency 10 \
  --max-tokens 256 \
  --temperature 0.7 \
  --capture-responses yes \
  --output-file results.jsonl \
  --summary-file summary.json \
  --timeout 30
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--base-url` | Yes | - | Base URL for the API endpoint |
| `--api-key` | No | api-key | API key for authorization |
| `--model` | Yes | - | Model identifier |
| `--length` | No | short | Prompt length (short/medium/long) |
| `--requests` | No | 1 | Total number of requests to send |
| `--concurrency` | No | 1 | Number of concurrent workers |
| `--max-tokens` | No | 128 | Maximum tokens to generate |
| `--temperature` | No | 0.0 | Sampling temperature |
| `--capture-responses` | No | no | Save responses to file (yes/no) |
| `--output-file` | No | responses.json | Output file for responses (JSONL format) |
| `--summary-file` | No | - | File to save summary metrics (JSON) |
| `--timeout` | No | 60.0 | Request timeout in seconds |

## Output Metrics

The tool provides comprehensive performance metrics:

- **Request Statistics**: Success/failure counts and rates
- **Latency Metrics**: 
  - Average latency (Time to First Byte)
  - P50 and P95 latency percentiles
  - Average round-trip time
- **Throughput Metrics**:
  - Requests per second
  - Input/output tokens per second
  - Tokens per second per request
  - Throughput per worker
- **Token Usage**: Total input and output tokens processed

### Example Output

```
Parameters:
+--------------------------------------------+
| base_url     : https://api.example.com     |
| api_key      : ****abcd                    |
| model        : gpt-3.5-turbo               |
| length       : medium                      |
| requests     : 50                          |
| concurrency  : 5                           |
| max_tokens   : 256                         |
| temperature  : 0.7                         |
+--------------------------------------------+

Test Results:
+-------------------------------------------+
| Successful requests     : 50              |
| Total requests          : 50              |
| Failures                : 0               |
| Total execution time    : 12.456 s        |
| Requests per second     : 4.02            |
| Avg latency (TTFB)      : 1247.3 ms       |
| p50 latency (TTFB)      : 1205.1 ms       |
| p95 latency (TTFB)      : 1456.8 ms       |
| Avg duration (RTT)      : 2491.2 ms       |
| Total input tokens      : 2500            |
| Total output tokens     : 12800           |
| Tokens/sec (output)     : 1027.84         |
| Tokens/sec (input)      : 200.64          |
| Tokens/sec per request  : 20.56           |
| Throughput tps/worker   : 205.57          |
+-------------------------------------------+

Last Response (Request #50):
+-----------------------------------------------------------------------------------+
| I'm doing well, thank you for asking! How can I assist you today? I'm here to... |
| help with any questions you might have about programming, writing, analysis, o... |
| r just about anything else you'd like to discuss.                                |
+-----------------------------------------------------------------------------------+
```

The CLI output now includes:
- **Comprehensive metrics table**: All performance statistics in an easy-to-read format
- **Last response preview**: Shows the content of the most recent successful response (truncated for readability)
- **Error details**: Failed requests are displayed in a structured table format

## Response Capture

When `--capture-responses yes` is used, all responses are saved in JSONL format with complete conversation context:

```json
{"request_id": 1, "ok": true, "status_code": 200, "latency_ms": 1247.3, "duration_ms": 2491.2, "input_tokens": 50, "output_tokens": 256, "request": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 128, "temperature": 0.7}, "response": {"choices": [{"message": {"role": "assistant", "content": "I'm doing well, thank you for asking!"}}]}, "error": null}
{"request_id": 2, "ok": true, "status_code": 200, "latency_ms": 1205.1, "duration_ms": 2387.5, "input_tokens": 50, "output_tokens": 244, "request": {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Hello, how are you?"}], "max_tokens": 128, "temperature": 0.7}, "response": {"choices": [{"message": {"role": "assistant", "content": "I'm doing great! How can I help you today?"}}]}, "error": null}
```

Each captured response includes:
- **Full request context**: The complete API request including model, messages, and parameters
- **Complete response**: The full API response with all metadata
- **Performance metrics**: Latency and duration measurements
- **Token usage**: Input and output token counts if available

## Summary Export

Use `--summary-file` to export metrics as JSON for integration with monitoring systems:

```json
{
  "successful_requests": 50,
  "total_requests": 50,
  "failures": 0,
  "total_time_s": 12.456,
  "requests_per_second": 4.02,
  "successful_rps": 4.02,
  "avg_latency_ms": 1247.3,
  "p50_latency_ms": 1205.1,
  "p95_latency_ms": 1456.8,
  "avg_duration_ms": 2491.2,
  "total_input_tokens": 2500,
  "total_output_tokens": 12800,
  "input_tokens_per_second": 200.64,
  "output_tokens_per_second": 1027.84,
  "tokens_per_second_per_request": 20.56,
  "throughput_tokens_per_second_per_worker": 205.57,
  "concurrency": 5
}
```

## Use Cases

- **Performance Testing**: Measure API response times and throughput
- **Load Testing**: Test server behavior under concurrent load
- **Capacity Planning**: Determine optimal concurrency levels
- **API Comparison**: Compare different models or providers
- **Monitoring**: Continuous performance monitoring in CI/CD pipelines

## Examples

### Test local development server
```bash
python benchmark.py --base-url http://localhost:8000 --model local-model --requests 10 --concurrency 2
```

### Stress test with high concurrency
```bash
python benchmark.py --base-url https://api.example.com --api-key $API_KEY --model gpt-4 --requests 200 --concurrency 20 --length long
```

### Benchmark different prompt lengths
```bash
# Short prompts
python benchmark.py --base-url $URL --api-key $KEY --model $MODEL --length short --requests 100 --summary-file short-results.json

# Medium prompts  
python benchmark.py --base-url $URL --api-key $KEY --model $MODEL --length medium --requests 100 --summary-file medium-results.json

# Long prompts
python benchmark.py --base-url $URL --api-key $KEY --model $MODEL --length long --requests 100 --summary-file long-results.json
```

## Requirements

- Python 3.7+
- tiktoken (for local token calculation)

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Token Calculation

The tool uses **tiktoken** for accurate local token calculation, providing several advantages:

- **API-Independent**: Token counts are calculated locally, not dependent on API responses
- **Consistent Metrics**: Get token statistics even for failed requests
- **Model-Aware**: Automatically selects the appropriate tokenizer for the specified model
- **Accurate Throughput**: Precise input/output token rates for performance analysis

### Supported Models

The tool automatically detects the appropriate tokenizer for:
- OpenAI models (GPT-3.5, GPT-4, etc.)
- Most OpenAI-compatible models
- Falls back to `cl100k_base` encoding for unknown models

Input tokens are calculated from the request messages, while output tokens are extracted from the API response content.

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions and support, please open an issue on the GitHub repository.
