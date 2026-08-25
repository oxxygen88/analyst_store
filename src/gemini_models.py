from __future__ import annotations

import importlib.util
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterable

# Stable Gemini model IDs verified against Google AI for Developers documentation.
GEMINI_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
GEMINI_DEFAULT_MODEL = GEMINI_MODELS[0]

GEMINI_MODEL_LABELS = {
    "gemini-3.7-flash": "Gemini 3.7 Flash (Recommended)",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite (Hemat)",
}


class GeminiModelUnavailableError(RuntimeError):
    """Raised when a Gemini model ID is unavailable/deprecated for this account."""


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def fallback_chain(selected_model: str) -> list[str]:
    """Return selected model plus same/lower-tier stable fallbacks.

    The chain never escalates from a cheaper user-selected model to a newer,
    potentially more expensive model. Unknown IDs get the standard stable list
    after the requested ID so old saved selections can self-heal.
    """
    selected = (selected_model or GEMINI_DEFAULT_MODEL).strip()
    if selected in GEMINI_MODELS:
        idx = GEMINI_MODELS.index(selected)
        return GEMINI_MODELS[idx:]
    return [selected] + [m for m in GEMINI_MODELS if m != selected]


def is_model_unavailable_error(exc_or_message) -> bool:
    text = str(exc_or_message or "").lower()
    markers = (
        "404",
        "not_found",
        "not found",
        "no longer available",
        "model is no longer available",
        "model not available",
        "model unavailable",
        "unsupported model",
        "does not exist",
        "is not supported",
    )
    # Avoid treating arbitrary 404 URLs as model errors unless Gemini/model wording exists.
    if "404" in text and ("model" in text or "models/" in text or "not_found" in text):
        return True
    return any(marker in text for marker in markers[1:])


def _extract_text(payload: dict) -> str:
    chunks = []
    for candidate in payload.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and part.get("text"):
                chunks.append(str(part["text"]))
    text = "\n".join(chunks).strip()
    if text:
        return text
    prompt_feedback = payload.get("promptFeedback") or {}
    block_reason = prompt_feedback.get("blockReason")
    if block_reason:
        raise RuntimeError(f"Gemini tidak menghasilkan teks karena request diblokir: {block_reason}.")
    raise RuntimeError("Gemini mengembalikan respons tanpa teks yang dapat dibaca.")


def _http_single(api_key: str, model: str, prompt: str, *, json_mode: bool) -> str:
    safe_model = urllib.parse.quote(model, safe="-._")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{safe_model}:generateContent"
    body_obj = {"contents": [{"parts": [{"text": prompt}]}]}
    if json_mode:
        # Gemini 3.x deprecates sampling params such as temperature/top_p/top_k.
        # Keep only the structured-output MIME request.
        body_obj["generationConfig"] = {"responseMimeType": "application/json"}
    body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
            err = parsed.get("error", {}) if isinstance(parsed, dict) else {}
            message = err.get("message") or raw
            status = err.get("status") or ""
        except Exception:
            message, status = raw, ""
        combined = f"HTTP {exc.code} {status}: {message}"
        if is_model_unavailable_error(combined):
            raise GeminiModelUnavailableError(combined) from exc
        if exc.code in (400, 401, 403):
            raise RuntimeError(
                f"Gemini API key/request ditolak ({exc.code}). Periksa API key Google AI Studio dan akses model. {message}"
            ) from exc
        if exc.code == 429:
            raise RuntimeError(
                f"Gemini API quota/limit tercapai (429). Periksa quota atau billing Gemini API. {message}"
            ) from exc
        raise RuntimeError(f"Gemini API error HTTP {exc.code} {status}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Tidak dapat terhubung ke Gemini API: {exc.reason}") from exc
    return _extract_text(payload)


def _sdk_single(api_key: str, model: str, prompt: str, *, json_mode: bool) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Google GenAI SDK tidak tersedia.") from exc

    try:
        client = genai.Client(api_key=api_key)
        kwargs = {"model": model, "contents": prompt}
        if json_mode:
            # Do not send deprecated Gemini 3.x sampling params.
            kwargs["config"] = types.GenerateContentConfig(response_mime_type="application/json")
        response = client.models.generate_content(**kwargs)
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise RuntimeError("Gemini mengembalikan respons tanpa teks yang dapat dibaca.")
        return text
    except Exception as exc:
        if is_model_unavailable_error(exc):
            raise GeminiModelUnavailableError(str(exc)) from exc
        # Keep quota/auth/network details from the SDK visible in a concise form.
        raise RuntimeError(f"Gemini API request gagal: {exc}") from exc


def call_gemini_with_fallback(
    api_key: str,
    model: str,
    prompt: str,
    *,
    json_mode: bool = False,
    prefer_sdk: bool = True,
) -> str:
    """Call Gemini and automatically fall back only when a model is unavailable.

    Authentication, quota, malformed-request, and network errors are NOT hidden by
    fallback because changing model would not solve them.
    """
    if not api_key:
        raise ValueError("Gemini API key belum diisi.")

    chain = fallback_chain(model)
    unavailable_errors: list[str] = []
    use_sdk = prefer_sdk and module_available("google.genai")

    for candidate in chain:
        try:
            if use_sdk:
                return _sdk_single(api_key, candidate, prompt, json_mode=json_mode)
            return _http_single(api_key, candidate, prompt, json_mode=json_mode)
        except GeminiModelUnavailableError as exc:
            unavailable_errors.append(f"{candidate}: {exc}")
            continue

    tried = " → ".join(chain)
    detail = unavailable_errors[-1] if unavailable_errors else "model tidak tersedia"
    raise RuntimeError(
        "Semua model Gemini fallback tidak tersedia untuk akun/API ini. "
        f"Model yang dicoba: {tried}. Detail terakhir: {detail}"
    )
