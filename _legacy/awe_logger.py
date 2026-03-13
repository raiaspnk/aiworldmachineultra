"""
===========================================================================
  AWE_LOGGER.PY – AI World Engine Structured Logger
===========================================================================

Sistema de log centralizado para toda a pipeline do AI World Engine.

Formato padronizado de saída:
  [HH:MM:SS] [MODULE  ] [PHASE    ] ● Mensagem
  [HH:MM:SS] [WORLD   ] [Chunk 0,0] ✔ RANSAC — 47ms
  [HH:MM:SS] [BRIDGE  ] [ETAPA 2/3] ✘ Erro crítico: ...

Módulos disponíveis:
  "BRIDGE"   — Orquestrador principal (bridge.py)
  "WORLD"    — World Generator (world_generator.py)
  "MONSTER"  — MonsterCore C++/CUDA (emitido via Python wrapper)
  "SAM3"     — Segmentação semântica
  "FAB"      — JSON Fab Spawner

Uso básico:
  from awe_logger import AWELogger
  log = AWELogger("WORLD")
  log.phase("Chunk 0,0")
  log.info("Processando terreno...")
  with log.timer("RANSAC"):
      ...   # código a medir
  log.ok("Chunk concluído")
  log.error("Algo deu errado", exc=e)
===========================================================================
"""

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── Cores ANSI (funciona em terminais Linux/SSH da GPU alugada) ─────────
class _C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"

# Mapa de cor por módulo
_MODULE_COLOR = {
    "BRIDGE":  _C.CYAN,
    "WORLD":   _C.GREEN,
    "MONSTER": _C.MAGENTA,
    "SAM3":    _C.YELLOW,
    "FAB":     _C.BLUE,
}

# ── Arquivo de log global (compartilhado entre todos os módulos) ─────────
_log_file_path: Optional[Path] = None
_log_file_handle = None

def configure_log_file(path: str):
    """
    Ativa gravação de logs em arquivo.
    Chamar uma vez em bridge.py antes de iniciar o pipeline.
    """
    global _log_file_path, _log_file_handle
    _log_file_path = Path(path)
    _log_file_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file_handle = open(str(_log_file_path), 'a', encoding='utf-8')


def _write_file(line: str):
    """Grava linha no arquivo sem ANSI."""
    if _log_file_handle:
        # Remover códigos ANSI para o arquivo de log
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', line)
        _log_file_handle.write(clean + '\n')
        _log_file_handle.flush()


# ── Logger principal ─────────────────────────────────────────────────────

class AWELogger:
    """
    Logger estruturado para um módulo específico da AI World Engine.
    """

    def __init__(self, module: str, phase: str = "INIT"):
        self.module = module.upper()
        self._phase = phase
        self._color = _MODULE_COLOR.get(self.module, _C.WHITE)

    def phase(self, phase_name: str) -> 'AWELogger':
        """
        Atualiza a fase atual. Retorna self para encadeamento.
        Ex: log.phase("Chunk 0,0").info("Processando...")
        """
        self._phase = phase_name
        return self

    def _format(self, level_sym: str, level_color: str, msg: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        mod = f"{self._color}{_C.BOLD}{self.module:<7}{_C.RESET}"
        ph  = f"{_C.DIM}{self._phase:<12}{_C.RESET}"
        sym = f"{level_color}{level_sym}{_C.RESET}"
        return f"{_C.GREY}[{ts}]{_C.RESET} [{mod}] [{ph}] {sym} {msg}"

    def info(self, msg: str):
        line = self._format("●", _C.WHITE, msg)
        print(line)
        _write_file(line)

    def ok(self, msg: str):
        line = self._format("✔", _C.GREEN, f"{_C.GREEN}{msg}{_C.RESET}")
        print(line)
        _write_file(line)

    def warn(self, msg: str):
        line = self._format("⚠", _C.YELLOW, f"{_C.YELLOW}{msg}{_C.RESET}")
        print(line)
        _write_file(line)

    def error(self, msg: str, exc: Optional[Exception] = None):
        line = self._format("✘", _C.RED, f"{_C.RED}{_C.BOLD}{msg}{_C.RESET}")
        print(line, file=sys.stderr)
        _write_file(line)
        if exc:
            import traceback
            tb = traceback.format_exc()
            tb_line = f"  {_C.DIM}{tb.strip()}{_C.RESET}"
            print(tb_line, file=sys.stderr)
            _write_file(tb_line)

    def step(self, current: int, total: int, msg: str):
        """Log de progresso com barra formatada: [2/5] Mensagem"""
        line = self._format("→", _C.CYAN, f"[{current}/{total}] {msg}")
        print(line)
        _write_file(line)

    def metric(self, label: str, value: str, unit: str = ""):
        """Log de métrica formatada: ⚡ RANSAC: 47ms"""
        line = self._format("⚡", _C.MAGENTA, f"{_C.BOLD}{label}:{_C.RESET} {_C.CYAN}{value}{unit}{_C.RESET}")
        print(line)
        _write_file(line)

    def separator(self, title: str = "", width: int = 60):
        """Imprime um separador visual com título."""
        if title:
            pad = (width - len(title) - 2) // 2
            line = f"{_C.DIM}{'─' * pad} {_C.BOLD}{title}{_C.RESET}{_C.DIM} {'─' * pad}{_C.RESET}"
        else:
            line = f"{_C.DIM}{'─' * width}{_C.RESET}"
        print(line)
        _write_file(line)

    @contextmanager
    def timer(self, label: str):
        """
        Context manager que mede e loga o tempo de um bloco.

        Uso:
          with log.timer("RANSAC"):
              resultado = gpu_ransac(...)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms < 1000:
                self.metric(label, f"{elapsed_ms:.1f}", "ms")
            else:
                self.metric(label, f"{elapsed_ms / 1000:.2f}", "s")


# ── Banner de início da pipeline ─────────────────────────────────────────

def print_pipeline_banner(prompt: str, style: str, scale: float):
    """Imprime o banner de abertura do pipeline."""
    w = 62
    bar = "═" * w
    prompt_short = prompt[:50] + ("..." if len(prompt) > 50 else "")
    lines = [
        f"\n{_C.CYAN}{_C.BOLD}╔{bar}╗",
        f"║{'AI WORLD ENGINE — PIPELINE INICIADA':^{w}}║",
        f"╠{bar}╣",
        f"║  Prompt : {prompt_short:<{w - 11}}║",
        f"║  Estilo : {style:<{w - 11}}║",
        f"║  Escala : {scale:<{w - 11}}║",
        f"╚{bar}╝{_C.RESET}\n",
    ]
    output = "\n".join(lines)
    print(output)
    _write_file(output)


def print_pipeline_summary(results: dict, total_seconds: float):
    """Imprime o resumo final do pipeline."""
    w = 62
    bar = "═" * w
    status = "✅ SUCESSO" if results.get("success") else "❌ FALHOU"
    color = _C.GREEN if results.get("success") else _C.RED

    lines = [
        f"\n{color}{_C.BOLD}╔{bar}╗",
        f"║{f'PIPELINE CONCLUÍDA — {status}':^{w}}║",
        f"╠{bar}╣",
        f"║  Tempo Total : {f'{total_seconds:.1f}s':<{w - 16}}║",
    ]

    if results.get("glb_path"):
        p = str(results["glb_path"])[-50:]
        lines.append(f"║  GLB         : {p:<{w - 16}}║")
    if results.get("world_json"):
        p = str(results["world_json"])[-50:]
        lines.append(f"║  JSON Spawner: {p:<{w - 16}}║")
    if results.get("error"):
        err = str(results["error"])[:50]
        lines.append(f"║  Erro        : {err:<{w - 16}}║")

    lines.append(f"╚{bar}╝{_C.RESET}\n")
    output = "\n".join(lines)
    print(output)
    _write_file(output)
