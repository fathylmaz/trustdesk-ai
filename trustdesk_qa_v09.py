"""TrustDesk AI v0.9: local-only live API contract and latency smoke tests.

Run the existing v0.6 unified API separately before executing this script.
Uses only Python standard library; never loads or modifies saved ML models.

All requests are restricted to 127.0.0.1 / localhost. Synthetic English
probes only. Latency is exploratory, not an independent performance benchmark.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

SAMPLES = (
    ("account_close", "I want to cancel my bank account."),
    ("lost_card", "My card was stolen yesterday."),
    ("out_of_domain", "How humid is the air in the city today?"),
)
MODELS = ("tfidf", "distilbert")


class QAError(RuntimeError):
    """Expected verification failure (helpful instead of an unhandled traceback)."""


def validate_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise QAError("Only local http://127.0.0.1 or http://localhost API URLs are allowed")
    if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise QAError("Pass only a plain local API base URL, e.g. http://127.0.0.1:8002")
    try:
        port = parsed.port
    except ValueError as exc:
        raise QAError("Invalid port in --url") from exc
    if port is None:
        raise QAError("Specify a local API port in --url")
    return f"http://{parsed.hostname}:{port}"


def request_json(base_url: str, path: str, method: str = "GET", payload: dict | None = None,
                 timeout: float = 30) -> tuple[int, dict, float]:
    if not path.startswith("/") or path.startswith("//") or method not in {"GET", "POST", "OPTIONS"}:
        raise QAError("Invalid local API request")
    base = validate_url(base_url)
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(base + path, data=body, method=method,
                  headers={"Content-Type": "application/json", "Accept": "application/json"})
    start = perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read()
    except HTTPError as exc:
        status = exc.code
        raw = exc.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise QAError(f"Local API not reachable ({base}). Start uvicorn on port 8002. Details: {exc}") from exc
    elapsed_ms = (perf_counter() - start) * 1000
    try:
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Response JSON is not an object")
    except (UnicodeError, ValueError) as exc:
        raise QAError(f"{method} {path} did not return a JSON object (HTTP {status})") from exc
    return status, data, elapsed_ms


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QAError(message)


def verify_prediction(item: dict, model_name: str) -> None:
    require(item.get("model") == model_name, f"Prediction model mismatch: expected {model_name}")
    require(isinstance(item.get("label"), str) and bool(item["label"]), "Missing prediction label")
    score = item.get("score")
    require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1,
            f"Invalid {model_name} score")
    require(item.get("automatic_routing_enabled") is False, "SECURITY FAIL: auto-routing enabled")
    require(isinstance(item.get("score_note"), str) and len(item["score_note"]) >= 10,
            "Missing score limitation note")
    duration = item.get("inference_ms")
    require(isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0,
            "Invalid model inference time")
    ranked = item.get("top_predictions")
    require(isinstance(ranked, list) and 1 <= len(ranked) <= 3, "Invalid top predictions")
    require(ranked[0].get("label") == item["label"], "Highest-ranked label mismatch")
    previous = float("inf")
    for rank in ranked:
        require(isinstance(rank, dict) and isinstance(rank.get("label"), str) and rank["label"],
                "Malformed ranked label")
        value = rank.get("score")
        require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
                "Malformed ranked score")
        require(value <= previous + 1e-9, "Predictions not sorted by score")
        previous = value
    require(abs(ranked[0]["score"] - score) < 1e-4, "Top score mismatch")


def nearest_rank(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile (no interpolation); consistent for small samples."""
    if not values or not (0 < percentile <= 100):
        raise ValueError("values must be nonempty and percentile in (0,100]")
    arr = sorted(values)
    return arr[max(0, math.ceil(percentile * len(arr) / 100) - 1)]


def latency_summary(wall_times: list[float], internal_times: list[float]) -> dict:
    if len(wall_times) != len(internal_times) or not wall_times:
        raise ValueError("Matching nonempty latency arrays required")
    return {
        "requests": len(wall_times),
        "round_trip_ms_mean": round(statistics.mean(wall_times), 3),
        "round_trip_ms_p50": round(nearest_rank(wall_times, 50), 3),
        "round_trip_ms_p95": round(nearest_rank(wall_times, 95), 3),
        "model_inference_ms_p50": round(nearest_rank(internal_times, 50), 3),
        "model_inference_ms_p95": round(nearest_rank(internal_times, 95), 3),
    }


def run(base_url: str, runs: int, warmup: int, timeout: float, output_file: Path) -> dict:
    base_url = validate_url(base_url)
    if not 3 <= runs <= 100 or not 0 <= warmup <= 20 or not 1 <= timeout <= 120:
        raise QAError("Require runs=3..100, warmup=0..20, timeout=1..120")
    status, health, _ = request_json(base_url, "/health", timeout=timeout)
    require(status == 200 and health.get("models_loaded") is True, "API /health did not confirm loaded models")
    require(health.get("automatic_routing_enabled") is False, "SECURITY FAIL: auto-routing enabled in /health")
    status, listing, _ = request_json(base_url, "/v1/models", timeout=timeout)
    require(status == 200, "GET /v1/models failed")
    require({x.get("id") for x in listing.get("models", [])} == set(MODELS), "Model IDs not as expected")

    cases = []
    for case_name, text in SAMPLES:
        status, data, elapsed_ms = request_json(base_url, "/v1/compare", "POST", {"text": text}, timeout)
        require(status == 200, f"Compare failed for {case_name}: HTTP {status}")
        require(data.get("text") == text, "API returned mismatched input text")
        require(isinstance(data.get("warning"), str) and "banking" in data["warning"].lower(),
                "Missing experiment limitations warning")
        preds = data.get("predictions")
        require(isinstance(preds, list) and len(preds) == 2, "Must return exactly two predictions")
        for item, name in zip(preds, MODELS):
            verify_prediction(item, name)
        cases.append({"case": case_name, "http_status": status, "round_trip_ms": round(elapsed_ms, 3),
                      "predictions": [{"model": item["model"], "label": item["label"],
                                       "inference_ms": item["inference_ms"]} for item in preds]})

    invalid_payloads = [({}, "/v1/compare"), ({"text": ""}, "/v1/compare"),
                        ({"text": "hi"}, "/v1/compare"),
                        ({"text": "a" * 2001}, "/v1/compare"),
                        ({"text": "Close my bank account.", "model": "invalid"}, "/v1/predict")]
    for payload, path in invalid_payloads:
        status, _, _ = request_json(base_url, path, "POST", payload, timeout)
        require(status == 422, f"Expected HTTP 422 for invalid request to {path}, got {status}")

    benchmarks = {}
    benchmark_text = "I want to cancel my bank account."  # fixed synthetic example, never user data
    for model in MODELS:
        for _ in range(warmup):
            status, item, _ = request_json(base_url, "/v1/predict", "POST",
                                          {"text": benchmark_text, "model": model}, timeout)
            require(status == 200, f"Warm-up request failed: {model}")
            verify_prediction(item, model)
        observed, internal = [], []
        for _ in range(runs):
            status, item, elapsed_ms = request_json(base_url, "/v1/predict", "POST",
                                                    {"text": benchmark_text, "model": model}, timeout)
            require(status == 200, f"Live benchmark request failed: {model}")
            verify_prediction(item, model)
            observed.append(elapsed_ms)
            internal.append(float(item["inference_ms"]))
        benchmarks[model] = latency_summary(observed, internal)

    report = {
        "kind": "TrustDesk AI v0.9 LOCAL EXPLORATORY API QA",
        "base_url": base_url,
        "health": health,
        "contract_checks": {"compare_scenarios_passed": len(SAMPLES),
                            "invalid_payloads_rejected": len(invalid_payloads),
                            "model_order": list(MODELS), "auto_routing_disabled": True},
        "scenarios": cases,
        "latency": benchmarks,
        "limitations": [
            "Synthetic sample requests only; no real customer messages are sent or stored.",
            "Test data and previously inspected validation results do NOT establish model accuracy, OOD safety or production readiness.",
            "Wall-clock RTT and API model inference time differ; run on same computer with models already loaded.",
            "Sequential requests only; this is NOT a load, concurrency, or scalability benchmark.",
            "p50/p95 are nearest-rank percentiles and can vary with GPU load, OS scheduling and number of samples.",
            "The server must remain localhost-only and automatic financial routing disabled.",
        ],
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="TrustDesk AI v0.9 local integration and exploratory latency QA")
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--runs", type=int, default=20, help="measured requests per model (3..100)")
    ap.add_argument("--warmup", type=int, default=3, help="warmup requests per model (0..20)")
    ap.add_argument("--timeout", type=float, default=30, help="request timeout seconds")
    ap.add_argument("--output", type=Path, default=Path("artifacts/qa_v09/report.json"))
    args = ap.parse_args()
    try:
        report = run(args.url, args.runs, args.warmup, args.timeout, args.output)
    except QAError as exc:
        print(f"QA FAILED: {exc}", file=sys.stderr)
        return 1
    print("\n=== TRUSTDESK v0.9 LOCAL API QA ===")
    print(json.dumps({"contract_checks": report["contract_checks"], "latency": report["latency"]},
                     indent=2, ensure_ascii=False))
    print(f"\nSaved report: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
