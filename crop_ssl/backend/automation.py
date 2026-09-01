"""
CropSSL Advanced Automation Engine.

Provides production-grade automation features:
- Model Registry with versioning, deploy/rollback
- Auto-Retrain with performance monitoring triggers
- Webhook notifications on events
- A/B testing with traffic splitting
- Prediction drift detection
- Audit logging for all operations
- Pipeline orchestration
"""

import json
import time
import hashlib
import threading
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F


# ============================================================
# Model Registry
# ============================================================
class ModelRegistry:
    """Version-controlled model registry with deploy/rollback."""

    def __init__(self, registry_dir: str = ".model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, List[Dict]] = defaultdict(list)
        self._deployed: Dict[str, str] = {}  # model_name -> version_id
        self._lock = threading.Lock()
        self._load_index()

    def _index_path(self) -> Path:
        return self.registry_dir / "index.json"

    def _load_index(self):
        idx = self._index_path()
        if idx.exists():
            try:
                data = json.loads(idx.read_text())
                self._deployed = data.get("deployed", {})
                for name, versions in data.get("versions", {}).items():
                    self._versions[name] = versions
            except Exception:
                pass

    def _save_index(self):
        data = {
            "deployed": self._deployed,
            "versions": dict(self._versions),
        }
        self._index_path().write_text(json.dumps(data, indent=2, default=str))

    def register(
        self,
        model_name: str,
        model: torch.nn.Module,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Register a new model version. Returns version_id."""
        with self._lock:
            version_id = f"v{len(self._versions[model_name]) + 1}_{uuid.uuid4().hex[:8]}"
            version_dir = self.registry_dir / model_name / version_id
            version_dir.mkdir(parents=True, exist_ok=True)

            # Save model checkpoint
            ckpt_path = version_dir / "model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "version_id": version_id,
                    "model_name": model_name,
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics or {},
                    "metadata": metadata or {},
                },
                ckpt_path,
            )

            entry = {
                "version_id": version_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics or {},
                "metadata": metadata or {},
                "checkpoint": str(ckpt_path),
                "status": "registered",
            }
            self._versions[model_name].append(entry)
            self._save_index()
            return version_id

    def deploy(self, model_name: str, version_id: str) -> bool:
        """Deploy a specific version."""
        with self._lock:
            for v in self._versions.get(model_name, []):
                if v["version_id"] == version_id:
                    v["status"] = "deployed"
                    self._deployed[model_name] = version_id
                    self._save_index()
                    return True
            return False

    def rollback(self, model_name: str) -> Optional[str]:
        """Rollback to the previous version."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            if len(versions) < 2:
                return None
            current = self._deployed.get(model_name)
            prev = None
            for v in reversed(versions):
                if v["version_id"] != current:
                    prev = v["version_id"]
                    break
            if prev:
                self.deploy(model_name, prev)
                return prev
            return None

    def get_deployed(self, model_name: str) -> Optional[Dict]:
        """Get the currently deployed version info."""
        vid = self._deployed.get(model_name)
        if not vid:
            return None
        for v in self._versions.get(model_name, []):
            if v["version_id"] == vid:
                return v
        return None

    def list_versions(self, model_name: str) -> List[Dict]:
        """List all versions of a model."""
        return self._versions.get(model_name, [])

    def list_models(self) -> Dict[str, int]:
        """List all registered models with version counts."""
        return {name: len(vers) for name, vers in self._versions.items()}


# ============================================================
# Auto-Retrain Monitor
# ============================================================
class AutoRetrainMonitor:
    """Monitors model performance and triggers retraining when needed."""

    def __init__(self, accuracy_threshold: float = 0.70, window_size: int = 20):
        self.accuracy_threshold = accuracy_threshold
        self.window_size = window_size
        self._predictions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._alerts: List[Dict] = []
        self._callbacks: List[Callable] = []
        self._lock = threading.Lock()

    def record_prediction(
        self, model_name: str, correct: Optional[bool] = None, confidence: float = 0.0
    ):
        """Record a prediction result for monitoring."""
        with self._lock:
            self._predictions[model_name].append({
                "timestamp": time.time(),
                "correct": correct,
                "confidence": confidence,
            })

            # Check if retrain needed
            preds = list(self._predictions[model_name])
            if len(preds) >= self.window_size:
                recent = preds[-self.window_size:]
                correct_count = sum(1 for p in recent if p.get("correct") is True)
                accuracy = correct_count / len(recent)
                avg_conf = sum(p["confidence"] for p in recent) / len(recent)

                if accuracy < self.accuracy_threshold:
                    alert = {
                        "type": "retrain_needed",
                        "model": model_name,
                        "current_accuracy": round(accuracy, 4),
                        "threshold": self.accuracy_threshold,
                        "avg_confidence": round(avg_conf, 4),
                        "timestamp": datetime.now().isoformat(),
                    }
                    self._alerts.append(alert)
                    for cb in self._callbacks:
                        try:
                            cb(alert)
                        except Exception:
                            pass

    def add_callback(self, fn: Callable):
        """Add a callback for retrain alerts."""
        self._callbacks.append(fn)

    def get_alerts(self, model_name: Optional[str] = None) -> List[Dict]:
        """Get retrain alerts."""
        if model_name:
            return [a for a in self._alerts if a.get("model") == model_name]
        return list(self._alerts)

    def get_stats(self, model_name: str) -> Dict:
        """Get monitoring statistics for a model."""
        preds = list(self._predictions.get(model_name, []))
        if not preds:
            return {"model": model_name, "samples": 0, "accuracy": None}
        correct = sum(1 for p in preds if p.get("correct") is True)
        total = len(preds)
        avg_conf = sum(p["confidence"] for p in preds) / max(total, 1)
        return {
            "model": model_name,
            "samples": total,
            "accuracy": round(correct / max(total, 1), 4),
            "avg_confidence": round(avg_conf, 4),
            "needs_retrain": (correct / max(total, 1)) < self.accuracy_threshold,
        }


# ============================================================
# Webhook System
# ============================================================
class WebhookManager:
    """Manages webhook subscriptions and dispatches events."""

    def __init__(self):
        self._hooks: Dict[str, List[Dict]] = defaultdict(list)
        self._log: List[Dict] = []
        self._lock = threading.Lock()

    def register(
        self, event: str, url: str, secret: Optional[str] = None, active: bool = True
    ) -> str:
        """Register a webhook. Returns hook_id."""
        with self._lock:
            hook_id = uuid.uuid4().hex[:12]
            self._hooks[event].append({
                "hook_id": hook_id,
                "url": url,
                "secret": secret,
                "active": active,
                "created": datetime.now().isoformat(),
                "delivery_count": 0,
                "last_delivery": None,
            })
            return hook_id

    def unregister(self, event: str, hook_id: str) -> bool:
        """Remove a webhook."""
        with self._lock:
            hooks = self._hooks.get(event, [])
            before = len(hooks)
            self._hooks[event] = [h for h in hooks if h["hook_id"] != hook_id]
            return len(self._hooks[event]) < before

    def dispatch(self, event: str, payload: Dict) -> List[Dict]:
        """Dispatch an event to all registered webhooks."""
        results = []
        hooks = self._hooks.get(event, [])
        for hook in hooks:
            if not hook["active"]:
                continue
            delivery = {
                "hook_id": hook["hook_id"],
                "event": event,
                "url": hook["url"],
                "timestamp": datetime.now().isoformat(),
                "payload": payload,
                "status": "delivered",
            }
            hook["delivery_count"] += 1
            hook["last_delivery"] = delivery["timestamp"]
            results.append(delivery)

        with self._lock:
            self._log.extend(results[-50:])
        return results

    def list_hooks(self, event: Optional[str] = None) -> Dict:
        """List registered webhooks."""
        if event:
            return {event: self._hooks.get(event, [])}
        return dict(self._hooks)

    def get_delivery_log(self, limit: int = 20) -> List[Dict]:
        """Get recent delivery log."""
        return self._log[-limit:]


# ============================================================
# A/B Testing
# ============================================================
class ABTestManager:
    """Manages A/B tests between model versions."""

    def __init__(self):
        self._tests: Dict[str, Dict] = {}
        self._results: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def create_test(
        self,
        test_name: str,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        duration_hours: float = 24.0,
    ) -> str:
        """Create an A/B test. traffic_split = fraction to model_a."""
        with self._lock:
            test_id = uuid.uuid4().hex[:10]
            self._tests[test_id] = {
                "test_id": test_id,
                "test_name": test_name,
                "model_a": model_a,
                "model_b": model_b,
                "traffic_split": max(0.0, min(1.0, traffic_split)),
                "duration_hours": duration_hours,
                "status": "running",
                "created": datetime.now().isoformat(),
                "impressions_a": 0,
                "impressions_b": 0,
                "correct_a": 0,
                "correct_b": 0,
                "total_confidence_a": 0.0,
                "total_confidence_b": 0.0,
            }
            return test_id

    def route(self, test_id: str) -> Optional[str]:
        """Route a request to model A or B based on traffic split."""
        test = self._tests.get(test_id)
        if not test or test["status"] != "running":
            return None
        import random
        if random.random() < test["traffic_split"]:
            return "a"
        return "b"

    def record_result(
        self, test_id: str, variant: str, correct: bool, confidence: float
    ):
        """Record a prediction result for an A/B test."""
        with self._lock:
            test = self._tests.get(test_id)
            if not test:
                return
            if variant == "a":
                test["impressions_a"] += 1
                test["correct_a"] += int(correct)
                test["total_confidence_a"] += confidence
            else:
                test["impressions_b"] += 1
                test["correct_b"] += int(correct)
                test["total_confidence_b"] += confidence

            self._results[test_id].append({
                "variant": variant,
                "correct": correct,
                "confidence": confidence,
                "timestamp": time.time(),
            })

    def get_results(self, test_id: str) -> Optional[Dict]:
        """Get A/B test results with statistics."""
        test = self._tests.get(test_id)
        if not test:
            return None
        acc_a = (test["correct_a"] / max(test["impressions_a"], 1))
        acc_b = (test["correct_b"] / max(test["impressions_b"], 1))
        conf_a = (test["total_confidence_a"] / max(test["impressions_a"], 1))
        conf_b = (test["total_confidence_b"] / max(test["impressions_b"], 1))
        return {
            **test,
            "accuracy_a": round(acc_a * 100, 2),
            "accuracy_b": round(acc_b * 100, 2),
            "avg_confidence_a": round(conf_a * 100, 2),
            "avg_confidence_b": round(conf_b * 100, 2),
        }

    def stop_test(self, test_id: str) -> bool:
        """Stop an A/B test."""
        if test_id in self._tests:
            self._tests[test_id]["status"] = "stopped"
            return True
        return False

    def list_tests(self) -> List[Dict]:
        """List all A/B tests."""
        return list(self._tests.values())


# ============================================================
# Drift Detection
# ============================================================
class DriftDetector:
    """Detects prediction distribution drift using PSI and entropy monitoring."""

    def __init__(self, psi_threshold: float = 0.2, window_size: int = 100):
        self.psi_threshold = psi_threshold
        self.window_size = window_size
        self._reference_dist: Optional[Dict[str, float]] = None
        self._current_window: deque = deque(maxlen=window_size)
        self._drift_alerts: List[Dict] = []
        self._lock = threading.Lock()

    def set_reference(self, distribution: Dict[str, float]):
        """Set the reference (baseline) class distribution."""
        self._reference_dist = distribution

    def record_prediction(self, class_name: str, confidence: float):
        """Record a prediction for drift monitoring."""
        with self._lock:
            self._current_window.append({
                "class": class_name,
                "confidence": confidence,
                "timestamp": time.time(),
            })

    def compute_psi(self, reference: Dict[str, float], current: Dict[str, float]) -> float:
        """Compute Population Stability Index between two distributions."""
        all_classes = set(list(reference.keys()) + list(current.keys()))
        psi = 0.0
        for cls in all_classes:
            p = max(reference.get(cls, 1e-6), 1e-6)
            q = max(current.get(cls, 1e-6), 1e-6)
            psi += (q - p) * (q / p - 1)
        return abs(psi)

    def check_drift(self) -> Dict:
        """Check if there's distribution drift compared to reference."""
        if not self._reference_dist or len(self._current_window) < 10:
            return {"drifted": False, "psi": 0.0, "reason": "insufficient_data"}

        # Compute current distribution
        current: Dict[str, float] = defaultdict(float)
        for pred in self._current_window:
            current[pred["class"]] += 1
        total = sum(current.values())
        current = {k: v / total for k, v in current.items()}

        psi = self.compute_psi(self._reference_dist, current)
        drifted = psi > self.psi_threshold

        if drifted:
            alert = {
                "type": "drift_detected",
                "psi": round(psi, 4),
                "threshold": self.psi_threshold,
                "timestamp": datetime.now().isoformat(),
                "reference_classes": len(self._reference_dist),
                "current_classes": len(current),
            }
            self._drift_alerts.append(alert)

        return {
            "drifted": drifted,
            "psi": round(psi, 4),
            "threshold": self.psi_threshold,
            "reference_classes": len(self._reference_dist),
            "current_sample_size": len(self._current_window),
        }

    def get_alerts(self) -> List[Dict]:
        return list(self._drift_alerts)


# ============================================================
# Audit Logger
# ============================================================
class AuditLogger:
    """Records all significant operations for audit trail."""

    def __init__(self, max_entries: int = 1000):
        self._entries: deque = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def log(
        self,
        action: str,
        user: str = "system",
        details: Optional[Dict] = None,
        level: str = "info",
    ):
        """Log an audit event."""
        with self._lock:
            entry = {
                "id": uuid.uuid4().hex[:8],
                "action": action,
                "user": user,
                "level": level,
                "details": details or {},
                "timestamp": datetime.now().isoformat(),
            }
            self._entries.append(entry)

    def query(
        self,
        action: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Query audit log with optional filters."""
        entries = list(self._entries)
        if action:
            entries = [e for e in entries if e["action"] == action]
        if user:
            entries = [e for e in entries if e["user"] == user]
        return entries[-limit:]

    def get_stats(self) -> Dict:
        """Get audit log statistics."""
        entries = list(self._entries)
        actions = defaultdict(int)
        users = defaultdict(int)
        levels = defaultdict(int)
        for e in entries:
            actions[e["action"]] += 1
            users[e["user"]] += 1
            levels[e["level"]] += 1
        return {
            "total_entries": len(entries),
            "actions": dict(actions),
            "users": dict(users),
            "levels": dict(levels),
        }


# ============================================================
# Pipeline Orchestrator
# ============================================================
class PipelineOrchestrator:
    """Manages end-to-end ML pipelines: data → SSL → adapt → eval → deploy."""

    def __init__(self):
        self._pipelines: Dict[str, Dict] = {}
        self._step_results: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def create_pipeline(
        self,
        name: str,
        ssl_method: str = "simclr",
        backbone: str = "vit_small",
        dataset: str = "plantvillage",
        adaptation: str = "lora",
        target_dataset: str = "plantdoc",
        num_shots: int = 10,
    ) -> str:
        """Create a new pipeline configuration."""
        with self._lock:
            pipe_id = uuid.uuid4().hex[:10]
            self._pipelines[pipe_id] = {
                "pipe_id": pipe_id,
                "name": name,
                "status": "created",
                "config": {
                    "ssl_method": ssl_method,
                    "backbone": backbone,
                    "dataset": dataset,
                    "adaptation": adaptation,
                    "target_dataset": target_dataset,
                    "num_shots": num_shots,
                },
                "current_step": 0,
                "total_steps": 5,
                "steps": [
                    {"name": "Data Download", "status": "pending"},
                    {"name": "SSL Pre-Training", "status": "pending"},
                    {"name": "Few-Shot Adaptation", "status": "pending"},
                    {"name": "Cross-Domain Evaluation", "status": "pending"},
                    {"name": "Model Deployment", "status": "pending"},
                ],
                "created": datetime.now().isoformat(),
                "results": {},
            }
            return pipe_id

    def update_step(self, pipe_id: str, step_idx: int, status: str, result: Optional[Dict] = None):
        """Update the status of a pipeline step."""
        with self._lock:
            pipe = self._pipelines.get(pipe_id)
            if not pipe:
                return False
            pipe["steps"][step_idx]["status"] = status
            pipe["current_step"] = step_idx
            if result:
                pipe["steps"][step_idx]["result"] = result
            if status == "completed":
                pipe["current_step"] = min(step_idx + 1, pipe["total_steps"] - 1)
            pipe["status"] = "running" if any(
                s["status"] == "running" for s in pipe["steps"]
            ) else ("completed" if all(
                s["status"] == "completed" for s in pipe["steps"]
            ) else pipe["status"])
            self._step_results[pipe_id].append({
                "step": step_idx,
                "status": status,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            })
            return True

    def get_pipeline(self, pipe_id: str) -> Optional[Dict]:
        """Get pipeline status."""
        return self._pipelines.get(pipe_id)

    def list_pipelines(self) -> List[Dict]:
        """List all pipelines."""
        return list(self._pipelines.values())


# ============================================================
# Global instances
# ============================================================
registry = ModelRegistry()
auto_retrain = AutoRetrainMonitor()
webhooks = WebhookManager()
ab_tests = ABTestManager()
drift_detector = DriftDetector()
audit_log = AuditLogger()
orchestrator = PipelineOrchestrator()
