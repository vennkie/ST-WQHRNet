from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from urllib import error, request

import yaml


@dataclass
class GroqConfig:
    api_key: str
    model: str
    temperature: float
    max_tokens: int


class GroqResearchService:
    def __init__(self, config_path: Path, eval_metrics_path: Path):
        self._config_path = config_path
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        groq_cfg = cfg.get("groq", {})
        api_key = os.getenv("GROQ_API_KEY", groq_cfg.get("api_key", "")).strip()

        self.config = GroqConfig(
            api_key=api_key,
            model=str(groq_cfg.get("model", "llama-3.1-8b-instant")),
            temperature=float(groq_cfg.get("temperature", 0.25)),
            max_tokens=int(groq_cfg.get("max_tokens", 1600)),
        )

        self.eval_metrics: Dict[str, Any] = {}
        if eval_metrics_path.exists():
            try:
                self.eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8"))
            except Exception:
                self.eval_metrics = {}

    def _reload_api_key(self) -> None:
        env_key = os.getenv("GROQ_API_KEY", "").strip()
        if env_key:
            self.config.api_key = env_key
            return

        try:
            with self._config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            file_key = str((cfg.get("groq", {}) or {}).get("api_key", "")).strip()
            if file_key:
                self.config.api_key = file_key
        except Exception:
            # Keep last known key if reload fails.
            pass

    @property
    def enabled(self) -> bool:
        self._reload_api_key()
        return bool(self.config.api_key)

    @property
    def key_fingerprint(self) -> str:
        self._reload_api_key()
        if not self.config.api_key:
            return "missing"
        key = self.config.api_key
        tail = key[-6:] if len(key) >= 6 else key
        return f"len={len(key)}:*{tail}"

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return {}
        return {}

    def _post_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._reload_api_key()
        if not self.config.api_key:
            raise RuntimeError("Groq API key is not configured.")

        url = "https://api.groq.com/openai/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "python-requests/2.31.0",
            },
        )

        try:
            with request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            detail = f"HTTP {exc.code}: {exc.reason}"
            if body:
                detail = f"{detail} | {body[:500]}"
            raise RuntimeError(f"Groq request failed: {detail}") from exc

    def auth_check(self) -> Dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "error": "Groq API key is not configured."}
        payload = {
            "model": self.config.model,
            "temperature": 0.0,
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": "Respond with one word: ok"},
                {"role": "user", "content": "healthcheck"},
            ],
        }
        try:
            self._post_chat(payload)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _eval_slice(self) -> Dict[str, Any]:
        return {
            "wqi_overall_acc": self.eval_metrics.get("wqi_overall_acc"),
            "wqi_macro_f1": self.eval_metrics.get("wqi_macro_f1"),
            "hhi_overall_acc": self.eval_metrics.get("hhi_overall_acc"),
            "hhi_macro_f1": self.eval_metrics.get("hhi_macro_f1"),
            "wqi_macro_roc_auc": self.eval_metrics.get("wqi_macro_roc_auc"),
            "hhi_macro_roc_auc": self.eval_metrics.get("hhi_macro_roc_auc"),
        }

    def generate_research_bundle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("Groq API key is not configured. Set GROQ_API_KEY or groq.api_key in config.yaml.")

        prompt_input = {
            "project": "Tamil Nadu Groundwater Risk Forecasting (ST-WQHRNet)",
            "base_forecast": context.get("base_forecast", {}),
            "scenario_forecast": context.get("scenario_forecast", {}),
            "comparison_forecast": context.get("comparison_forecast", {}),
            "management_plan": context.get("management_plan", {}),
            "analytics": context.get("analytics", {}),
            "evaluation_metrics": self._eval_slice(),
            "provenance": [
                "model output via /api/predict",
                "deterministic scenario engine in app.py",
                "eval metrics from outputs/eval_metrics.json",
            ],
        }

        system = (
            "You are a groundwater-risk research co-pilot for public policy teams. "
            "Return strict JSON only. No markdown, no commentary."
        )
        user = (
            "Create a research dashboard intelligence bundle with exactly these top-level keys: "
            "insight_panel, scenario_analysis, uncertainty_narrative, district_comparison, management_strategies, journal_summary. "
            "Each key must be an object. Keep claims tightly grounded to given inputs and metrics only. "
            "If an inference is weak, explicitly mark it low-confidence. Input:\n"
            + json.dumps(prompt_input, ensure_ascii=False)
        )

        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        raw = self._post_chat(payload)
        content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")

        parsed = self._extract_json(content)
        if not isinstance(parsed, dict):
            parsed = {}

        required = [
            "insight_panel",
            "scenario_analysis",
            "uncertainty_narrative",
            "district_comparison",
            "management_strategies",
            "journal_summary",
        ]
        for key in required:
            if key not in parsed or not isinstance(parsed.get(key), dict):
                parsed[key] = {"note": "No content generated."}

        parsed["meta"] = {
            "model": self.config.model,
            "provider": "Groq",
        }
        return parsed

    def chat_with_context(self, context: Dict[str, Any], question: str) -> str:
        if not self.enabled:
            raise RuntimeError("Groq API key is not configured.")

        system = (
            "You are a technical groundwater risk analyst assistant. "
            "Respond concisely, grounded in supplied context and model outputs. "
            "If context is insufficient, state the limitation clearly."
        )
        user = (
            "Answer the user question using this context only. "
            "Context JSON:\n"
            + json.dumps(context, ensure_ascii=False)
            + "\n\nQuestion: "
            + str(question)
        )

        payload = {
            "model": self.config.model,
            "temperature": 0.2,
            "max_tokens": 700,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        raw = self._post_chat(payload)
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return text or "No response generated."
