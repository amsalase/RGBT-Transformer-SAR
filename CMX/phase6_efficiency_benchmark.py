#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FASE 6 — EFICIENCIA COMPUTACIONAL RGB-T SAR

Mide sobre UNA GPU y batch=1:
- Parámetros totales y entrenables.
- Tamaño teórico de parámetros.
- FLOPs (si fvcore está disponible; se reporta backend y ops no soportadas).
- VRAM CUDA pico.
- Latencia GPU por forward mediante CUDA Events.
- Latencia wall-clock sincronizada.
- FPS.
- FP32 y/o FP16.

IMPORTANTE:
- Este benchmark mide el FORWARD DIRECTO de la red EncoderDecoder.
- NO mide sliding-window, multi-scale, flip, lectura de disco ni preprocesado.
- Usa la misma construcción de modelo que eval_phase4.py:
      EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
- Para reproducibilidad se recomienda cargar el checkpoint CORREGIDO.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from config import config
from models.builder import EncoderDecoder as SegModel


MIB = 1024 ** 2
GIB = 1024 ** 3


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def fmt_float(x, digits=4):
    if x is None:
        return "N/A"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(x):
        return "N/A"
    return f"{x:.{digits}f}"


def run_cmd(cmd):
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return {
            "command": " ".join(cmd),
            "returncode": int(p.returncode),
            "output": p.stdout,
        }
    except Exception as e:
        return {
            "command": " ".join(cmd),
            "returncode": -1,
            "output": repr(e),
        }


def cuda_info(device_index=0):
    props = torch.cuda.get_device_properties(device_index)
    return {
        "device_index_logical": int(device_index),
        "name": torch.cuda.get_device_name(device_index),
        "total_memory_gib": props.total_memory / GIB,
        "compute_capability": f"{props.major}.{props.minor}",
        "multi_processor_count": getattr(props, "multi_processor_count", None),
    }


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value, key

        # Un checkpoint puede ser directamente un state_dict.
        if obj and all(isinstance(k, str) for k in obj.keys()):
            tensor_values = sum(
                1 for v in obj.values()
                if torch.is_tensor(v)
            )
            if tensor_values >= max(1, len(obj) // 2):
                return obj, "<root>"

    raise RuntimeError(
        "No se pudo identificar un state_dict en el checkpoint. "
        "Se esperaban claves model/state_dict/model_state_dict o un state_dict raíz."
    )


def normalize_state_dict_keys(state):
    out = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = value
    return out


def load_checkpoint_flexible(model, checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"No existe checkpoint: {checkpoint}")

    obj = torch.load(
        str(checkpoint),
        map_location="cpu"
    )

    state, state_key = extract_state_dict(obj)
    state = normalize_state_dict_keys(state)

    result = model.load_state_dict(
        state,
        strict=False
    )

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_size_mib": checkpoint.stat().st_size / MIB,
        "state_dict_location": state_key,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }

    # Para un benchmark del modelo correcto no queremos que se haya cargado
    # un checkpoint claramente incompatible.
    total_model_keys = len(model.state_dict())
    bad_fraction = (
        (len(missing) + len(unexpected))
        / max(1, total_model_keys)
    )

    meta["key_mismatch_fraction_vs_model_keys"] = bad_fraction

    if bad_fraction > 0.05:
        raise RuntimeError(
            "El checkpoint parece incompatible con el modelo actual: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"model_keys={total_model_keys}. "
            "Se detiene para no medir una arquitectura/checkpoint incorrectos."
        )

    del obj
    return meta


def infer_input_hw(args) -> Tuple[int, int, str]:
    if args.input_height is not None or args.input_width is not None:
        if args.input_height is None or args.input_width is None:
            raise ValueError(
                "Debes especificar conjuntamente --input-height y --input-width."
            )
        return int(args.input_height), int(args.input_width), "CLI"

    crop = getattr(config, "eval_crop_size", None)

    if crop is None:
        raise RuntimeError(
            "config.eval_crop_size no existe. "
            "Define --input-height y --input-width explícitamente."
        )

    if isinstance(crop, int):
        return int(crop), int(crop), "config.eval_crop_size"

    if isinstance(crop, (list, tuple)) and len(crop) >= 2:
        return int(crop[0]), int(crop[1]), "config.eval_crop_size"

    raise RuntimeError(
        f"No pude interpretar config.eval_crop_size={crop!r}. "
        "Usa --input-height y --input-width."
    )


def infer_modal_channels(model):
    # CMX/SegFormer dual suele exponer extra_patch_embed1.proj.
    candidates = [
        "backbone.extra_patch_embed1.proj",
        "backbone.extra_patch_embed1",
    ]

    modules = dict(model.named_modules())

    for name in candidates:
        module = modules.get(name)
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels), name
        if module is not None:
            for child_name, child in module.named_modules():
                if isinstance(child, nn.Conv2d):
                    return int(child.in_channels), f"{name}.{child_name}".rstrip(".")

    # Fallback: primer Conv2d cuyo nombre sugiera rama extra/modal.
    for name, module in model.named_modules():
        low = name.lower()
        if isinstance(module, nn.Conv2d) and (
            "extra" in low or "modal" in low or "thermal" in low
        ):
            return int(module.in_channels), name

    # Fallback conservador para CMX.
    return 3, "<fallback=3>"


def parameter_stats(model):
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    buffers = sum(int(b.numel()) for b in model.buffers())

    return {
        "params_total": total,
        "params_trainable": trainable,
        "buffers_total": buffers,
        "params_million": total / 1e6,
        "trainable_million": trainable / 1e6,
        "weights_fp32_mib_theoretical": total * 4 / MIB,
        "weights_fp16_mib_theoretical": total * 2 / MIB,
    }
def attempt_fvcore_flops(model, rgb, modal_x):
    from collections import Counter
    from math import prod

    result = {
        "available": False,
        "backend": None,
        "total_flops": None,
        "gflops": None,
        "unsupported_ops": {},
        "uncalled_modules_count": None,
        "uncalled_modules": [],
        "error": None,
        "convention_note": (
            "fvcore + custom handlers. "
            "MAC/FMA follows fvcore convention. "
            "All previously unsupported operators are explicitly handled."
        ),
    }

    try:
        from fvcore.nn import FlopCountAnalysis
        from fvcore.nn.jit_handles import get_shape

    except Exception as e:
        result["error"] = (
            "fvcore no disponible: " + repr(e)
        )
        return result

    # ==========================================================
    # FUNCIONES AUXILIARES
    # ==========================================================

    def numel_from_value(value):
        shape = get_shape(value)

        if shape is None:
            return 0

        return int(prod(shape))


    # ==========================================================
    # 1. MUL
    #
    # Una multiplicacion elemento a elemento = 1 FLOP
    # por elemento de salida.
    # ==========================================================

    def mul_flop_jit(inputs, outputs):
        n = numel_from_value(outputs[0])

        return Counter({
            "mul": n
        })


    # ==========================================================
    # 2. ADD
    #
    # Una suma elemento a elemento = 1 FLOP
    # por elemento de salida.
    # ==========================================================

    def add_flop_jit(inputs, outputs):
        n = numel_from_value(outputs[0])

        return Counter({
            "add": n
        })


    # ==========================================================
    # 3. UPSAMPLE LINEAR 1D
    #
    # Interpolacion lineal:
    #
    # y = a + w * (b - a)
    #
    # Con la convencion FMA/MAC de fvcore:
    #
    # (b-a)  -> 1
    # FMA    -> 1
    #
    # Total aproximado estructural = 2 FLOPs/output
    # ==========================================================

    def upsample_linear1d_flop_jit(inputs, outputs):
        n = numel_from_value(outputs[0])

        return Counter({
            "upsample_linear1d": 2 * n
        })


    # ==========================================================
    # 4. SOFTMAX
    #
    # Para un vector de longitud L:
    #
    # x - max(x)        -> L
    # exp(x)            -> L
    # suma              -> L-1
    # division          -> L
    #
    # Las comparaciones utilizadas para obtener max()
    # NO se consideran FLOPs.
    #
    # Total:
    #
    # 4L - 1 por vector
    # ==========================================================

    def softmax_flop_jit(inputs, outputs):
        input_shape = get_shape(inputs[0])

        if input_shape is None:
            return Counter({"softmax": 0})

        n_total = int(prod(input_shape))

        # aten::softmax(input, dim, dtype)
        dim = inputs[1].toIValue()

        if dim is None:
            return Counter({
                "softmax": 4 * n_total
            })

        dim = int(dim)

        if dim < 0:
            dim += len(input_shape)

        length = int(input_shape[dim])

        if length <= 0:
            return Counter({"softmax": 0})

        n_vectors = n_total // length

        flops = (
            4 * n_total
            - n_vectors
        )

        return Counter({
            "softmax": flops
        })


    # ==========================================================
    # 5. GELU
    #
    # PyTorch puede usar:
    #
    # approximate="none"
    # approximate="tanh"
    #
    # Adoptamos una convencion explicita por elemento.
    #
    # exact:
    # 0.5*x*(1+erf(x/sqrt(2)))
    #
    # Se contabiliza erf como una operacion especial.
    #
    # ~5 operaciones por elemento.
    #
    # tanh approximation:
    # ~9 operaciones por elemento.
    # ==========================================================

    def gelu_flop_jit(inputs, outputs):
        n = numel_from_value(outputs[0])

        approximate = "none"

        if len(inputs) > 1:
            try:
                value = inputs[1].toIValue()

                if value is not None:
                    approximate = str(value)

            except Exception:
                pass

        if approximate == "tanh":
            flops_per_element = 9
        else:
            flops_per_element = 5

        return Counter({
            "gelu": flops_per_element * n
        })


    # ==========================================================
    # 6. SIGMOID
    #
    # sigmoid(x) = 1 / (1 + exp(-x))
    #
    # Convencion:
    #
    # negacion       -> 1
    # exp            -> 1
    # suma           -> 1
    # division       -> 1
    #
    # = 4 operaciones por elemento.
    # ==========================================================

    def sigmoid_flop_jit(inputs, outputs):
        n = numel_from_value(outputs[0])

        return Counter({
            "sigmoid": 4 * n
        })


    # ==========================================================
    # 7. ADAPTIVE MAX POOL 2D
    #
    # Max-pooling realiza comparaciones, no operaciones
    # aritmeticas floating-point.
    #
    # Por definicion FLOP:
    #
    # = 0 FLOPs
    #
    # Sigue marcado como HANDLED para que no aparezca
    # en unsupported_ops.
    # ==========================================================

    def adaptive_max_pool2d_flop_jit(inputs, outputs):
        return Counter({
            "adaptive_max_pool2d": 0
        })


    # ==========================================================
    # 8. FEATURE DROPOUT
    #
    # El modelo esta en model.eval().
    # Dropout es identidad durante inferencia.
    #
    # = 0 FLOPs
    # ==========================================================

    def feature_dropout_flop_jit(inputs, outputs):
        return Counter({
            "feature_dropout": 0
        })


    # ==========================================================
    # CREAR ANALISIS
    # ==========================================================

    try:
        result["available"] = True

        result["backend"] = (
            "fvcore.nn.FlopCountAnalysis + custom handlers"
        )

        analysis = FlopCountAnalysis(
            model,
            (rgb, modal_x)
        )

        # ======================================================
        # REGISTRAR OPERACIONES QUE FVCORE NO CONOCE
        # ======================================================

        analysis = analysis.set_op_handle(
            "aten::mul",
            mul_flop_jit
        )

        analysis = analysis.set_op_handle(
            "aten::add",
            add_flop_jit
        )

        analysis = analysis.set_op_handle(
            "aten::upsample_linear1d",
            upsample_linear1d_flop_jit
        )

        analysis = analysis.set_op_handle(
            "aten::softmax",
            softmax_flop_jit
        )

        analysis = analysis.set_op_handle(
            "aten::gelu",
            gelu_flop_jit
        )

        analysis = analysis.set_op_handle(
            "aten::sigmoid",
            sigmoid_flop_jit
        )

        analysis = analysis.set_op_handle(
            "aten::adaptive_max_pool2d",
            adaptive_max_pool2d_flop_jit
        )

        analysis = analysis.set_op_handle(
            "aten::feature_dropout",
            feature_dropout_flop_jit
        )

        # ======================================================
        # CALCULAR
        # ======================================================

        with torch.inference_mode():

            total = float(
                analysis.total()
            )

            unsupported = (
                analysis.unsupported_ops()
            )

            uncalled = (
                analysis.uncalled_modules()
            )

            by_operator = (
                analysis.by_operator()
            )

        result["total_flops"] = total

        result["gflops"] = (
            total / 1e9
        )

        result["unsupported_ops"] = {
            str(k): int(v)
            for k, v in unsupported.items()
        }

        result["uncalled_modules_count"] = (
            len(uncalled)
        )

        result["uncalled_modules"] = sorted(
            str(x)
            for x in uncalled
        )

        result["by_operator"] = {
            str(k): float(v)
            for k, v in by_operator.items()
        }

    except Exception as e:

        result["error"] = repr(e)

    return result

def create_inputs(
    height,
    width,
    modal_channels,
    device,
    dtype,
):
    rgb = torch.randn(
        1, 3, height, width,
        device=device,
        dtype=dtype
    )

    modal_x = torch.randn(
        1, modal_channels, height, width,
        device=device,
        dtype=dtype
    )

    return rgb, modal_x


def percentile(values, p):
    if not values:
        return math.nan
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def latency_stats(samples_ms):
    vals = [float(x) for x in samples_ms]
    return {
        "iterations": len(vals),
        "latency_event_mean_ms": statistics.mean(vals),
        "latency_event_median_ms": statistics.median(vals),
        "latency_event_std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "latency_event_min_ms": min(vals),
        "latency_event_max_ms": max(vals),
        "latency_event_p05_ms": percentile(vals, 0.05),
        "latency_event_p95_ms": percentile(vals, 0.95),
        "fps_from_event_mean": 1000.0 / statistics.mean(vals),
        "fps_from_event_median": 1000.0 / statistics.median(vals),
    }


def benchmark_precision(
    base_model_cpu,
    checkpoint_path,
    precision,
    height,
    width,
    modal_channels,
    warmup,
    repeats,
    device,
):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Construimos un modelo independiente por precisión para que FP16 sí
    # reduzca el peso de parámetros en GPU.
    model = SegModel(
        cfg=config,
        criterion=None,
        norm_layer=nn.BatchNorm2d
    )

    checkpoint_meta = None
    if checkpoint_path is not None:
        checkpoint_meta = load_checkpoint_flexible(
            model,
            checkpoint_path
        )

    model.eval()

    if precision == "fp16":
        model = model.half()
        dtype = torch.float16
    elif precision == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(precision)

    model = model.to(device)

    rgb, modal_x = create_inputs(
        height,
        width,
        modal_channels,
        device,
        dtype,
    )

    torch.cuda.synchronize(device)

    allocated_after_setup = torch.cuda.memory_allocated(device)
    reserved_after_setup = torch.cuda.memory_reserved(device)

    # Dry run: además valida canales/forma.
    with torch.inference_mode():
        out = model(rgb, modal_x)
    torch.cuda.synchronize(device)

    if isinstance(out, (tuple, list)):
        out_shape = [list(x.shape) for x in out if torch.is_tensor(x)]
    elif torch.is_tensor(out):
        out_shape = list(out.shape)
    else:
        out_shape = str(type(out))

    del out

    # Warm-up.
    with torch.inference_mode():
        for _ in range(warmup):
            out = model(rgb, modal_x)
        del out

    torch.cuda.synchronize(device)

    # El pico se resetea DESPUÉS del warm-up, manteniendo como baseline
    # las asignaciones actuales (modelo + inputs).
    torch.cuda.reset_peak_memory_stats(device)

    samples_ms = []

    with torch.inference_mode():
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            out = model(rgb, modal_x)
            end.record()

            end.synchronize()
            samples_ms.append(
                float(start.elapsed_time(end))
            )

            del out

    torch.cuda.synchronize(device)

    # Wall-clock sincronizado para throughput sostenido.
    torch.cuda.synchronize(device)
    wall_t0 = time.perf_counter()

    with torch.inference_mode():
        for _ in range(repeats):
            out = model(rgb, modal_x)
        del out

    torch.cuda.synchronize(device)
    wall_t1 = time.perf_counter()

    wall_total_s = wall_t1 - wall_t0
    wall_mean_ms = wall_total_s * 1000.0 / repeats
    wall_fps = repeats / wall_total_s

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    stats = latency_stats(samples_ms)

    stats.update({
        "precision": precision,
        "batch_size": 1,
        "input_height": height,
        "input_width": width,
        "rgb_channels": 3,
        "modal_channels": modal_channels,
        "warmup_iterations": warmup,
        "measurement_iterations": repeats,
        "output_shape": out_shape,
        "cuda_allocated_after_model_inputs_mib": allocated_after_setup / MIB,
        "cuda_reserved_after_model_inputs_mib": reserved_after_setup / MIB,
        "cuda_peak_allocated_mib": peak_allocated / MIB,
        "cuda_peak_reserved_mib": peak_reserved / MIB,
        "wall_mean_ms": wall_mean_ms,
        "fps_wall_clock": wall_fps,
        "checkpoint_loaded": checkpoint_meta is not None,
    })

    # Guardar muestras antes de liberar.
    raw_samples = samples_ms[:]

    del model, rgb, modal_x
    gc.collect()
    torch.cuda.empty_cache()

    return stats, raw_samples, checkpoint_meta


def write_csv(path: Path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({
                k: row.get(k, "")
                for k in fields
            })


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output",
        type=Path,
        required=True
    )

    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint CORREGIDO. Muy recomendado."
    )

    ap.add_argument(
        "--allow-no-checkpoint",
        action="store_true",
        help="Permite medir arquitectura sin cargar pesos. No recomendado para el registro final."
    )

    ap.add_argument(
        "--input-height",
        type=int,
        default=None
    )

    ap.add_argument(
        "--input-width",
        type=int,
        default=None
    )

    ap.add_argument(
        "--modal-channels",
        type=int,
        default=None,
        help="Si se omite, se intenta inferir del modelo."
    )

    ap.add_argument(
        "--warmup",
        type=int,
        default=30
    )

    ap.add_argument(
        "--repeats",
        type=int,
        default=100
    )

    ap.add_argument(
        "--precision",
        choices=("fp32", "fp16", "both"),
        default="both"
    )

    ap.add_argument(
        "--miou",
        type=float,
        default=None,
        help="mIoU (%) ya validado de Fase 4; solo para integrar la tabla resumen."
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. La Fase 6 debe ejecutarse sobre GPU."
        )

    if args.checkpoint is None and not args.allow_no_checkpoint:
        raise RuntimeError(
            "No se indicó --checkpoint. "
            "Para el registro final usa el checkpoint CORREGIDO. "
            "Si solo quieres una prueba de arquitectura, usa --allow-no-checkpoint."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Benchmark settings.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    height, width, hw_source = infer_input_hw(args)

    # Modelo CPU temporal para parámetros, canales y FLOPs.
    base_model = SegModel(
        cfg=config,
        criterion=None,
        norm_layer=nn.BatchNorm2d
    )
    base_model.eval()

    checkpoint_meta_complexity = None
    if args.checkpoint is not None:
        checkpoint_meta_complexity = load_checkpoint_flexible(
            base_model,
            args.checkpoint
        )

    pstats = parameter_stats(base_model)

    inferred_modal_channels, channel_source = infer_modal_channels(
        base_model
    )

    modal_channels = (
        int(args.modal_channels)
        if args.modal_channels is not None
        else inferred_modal_channels
    )

    # Para FLOPs usamos CUDA para evitar una traza CPU muy costosa.
    base_model = base_model.to(device)
    rgb_flops, x_flops = create_inputs(
        height,
        width,
        modal_channels,
        device,
        torch.float32
    )

    flops_info = attempt_fvcore_flops(
        base_model,
        rgb_flops,
        x_flops
    )

    del base_model, rgb_flops, x_flops
    gc.collect()
    torch.cuda.empty_cache()

    precisions = (
        ["fp32", "fp16"]
        if args.precision == "both"
        else [args.precision]
    )

    benchmark_rows = []
    checkpoint_reports = {}

    for precision in precisions:
        try:
            row, samples, cp_meta = benchmark_precision(
                base_model_cpu=None,
                checkpoint_path=args.checkpoint,
                precision=precision,
                height=height,
                width=width,
                modal_channels=modal_channels,
                warmup=args.warmup,
                repeats=args.repeats,
                device=device,
            )

            benchmark_rows.append(row)
            checkpoint_reports[precision] = cp_meta

            sample_rows = [
                {
                    "iteration": i + 1,
                    "latency_ms": value
                }
                for i, value in enumerate(samples)
            ]

            write_csv(
                args.output / f"latency_samples_{precision}.csv",
                sample_rows,
                ["iteration", "latency_ms"]
            )

        except Exception as e:
            benchmark_rows.append({
                "precision": precision,
                "status": "FAILED",
                "error": repr(e),
                "batch_size": 1,
                "input_height": height,
                "input_width": width,
            })

    # Completar datos comunes.
    for row in benchmark_rows:
        row.setdefault("status", "OK")
        row["params_million"] = pstats["params_million"]
        row["gflops"] = flops_info.get("gflops")
        row["miou_percent"] = args.miou
        row["gpu_name"] = torch.cuda.get_device_name(0)

    fields = [
        "precision",
        "status",
        "batch_size",
        "input_height",
        "input_width",
        "params_million",
        "gflops",
        "cuda_peak_allocated_mib",
        "cuda_peak_reserved_mib",
        "latency_event_mean_ms",
        "latency_event_median_ms",
        "latency_event_std_ms",
        "latency_event_p05_ms",
        "latency_event_p95_ms",
        "fps_from_event_mean",
        "wall_mean_ms",
        "fps_wall_clock",
        "miou_percent",
        "gpu_name",
        "error",
    ]

    write_csv(
        args.output / "efficiency_summary.csv",
        benchmark_rows,
        fields
    )

    environment = {
        "captured_at": now_iso(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": cuda_info(0),
        "nvidia_smi": run_cmd(["nvidia-smi"]),
    }

    complexity = {
        "input": {
            "batch_size": 1,
            "height": height,
            "width": width,
            "size_source": hw_source,
            "rgb_channels": 3,
            "modal_channels": modal_channels,
            "modal_channels_source": (
                "CLI"
                if args.modal_channels is not None
                else channel_source
            ),
        },
        "parameters": pstats,
        "flops": flops_info,
        "checkpoint": checkpoint_meta_complexity,
    }

    (args.output / "model_complexity.json").write_text(
        json.dumps(
            complexity,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    (args.output / "environment.json").write_text(
        json.dumps(
            environment,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    (args.output / "benchmark_details.json").write_text(
        json.dumps(
            {
                "benchmark_rows": benchmark_rows,
                "checkpoint_loads": checkpoint_reports,
                "protocol": {
                    "direct_model_forward": True,
                    "includes_disk_io": False,
                    "includes_preprocessing": False,
                    "includes_sliding_window": False,
                    "includes_multiscale": False,
                    "includes_flip": False,
                    "batch_size": 1,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "cuda_events": True,
                    "wall_clock_synchronized": True,
                    "cudnn_benchmark": True,
                },
            },
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )

    # REPORT
    lines = [
        "SAR — FASE 6: EFICIENCIA COMPUTACIONAL",
        "=" * 88,
        f"Fecha                         : {now_iso()}",
        f"GPU                           : {torch.cuda.get_device_name(0)}",
        f"PyTorch                       : {torch.__version__}",
        f"CUDA PyTorch                  : {torch.version.cuda}",
        f"cuDNN                         : {torch.backends.cudnn.version()}",
        f"CUDA_VISIBLE_DEVICES          : {os.environ.get('CUDA_VISIBLE_DEVICES')}",
        "",
        "PROTOCOLO",
        f"  Batch size                  : 1",
        f"  Input RGB                   : 1 x 3 x {height} x {width}",
        f"  Input modal                 : 1 x {modal_channels} x {height} x {width}",
        f"  Fuente tamaño               : {hw_source}",
        f"  Warm-up                     : {args.warmup}",
        f"  Repeticiones                : {args.repeats}",
        "  Latencia principal          : CUDA Events, forward directo",
        "  Wall-clock                  : sincronizado",
        "  I/O de disco                : NO",
        "  Preprocesado                : NO",
        "  Sliding-window              : NO",
        "  Multi-scale / flip          : NO",
        "",
        "COMPLEJIDAD",
        f"  Params total                : {pstats['params_total']}",
        f"  Params (M)                  : {pstats['params_million']:.4f}",
        f"  Params entrenables (M)      : {pstats['trainable_million']:.4f}",
        f"  Peso teórico FP32 (MiB)     : {pstats['weights_fp32_mib_theoretical']:.2f}",
        f"  Peso teórico FP16 (MiB)     : {pstats['weights_fp16_mib_theoretical']:.2f}",
        f"  FLOPs backend               : {flops_info.get('backend') or 'N/A'}",
        f"  GFLOPs                      : {fmt_float(flops_info.get('gflops'), 4)}",
        f"  FLOPs error                 : {flops_info.get('error') or 'NONE'}",
        f"  Unsupported FLOPs ops       : {flops_info.get('unsupported_ops') or {}}",
        "",
    ]

    if args.miou is not None:
        lines.append(f"mIoU Fase 4 aportado           : {args.miou:.2f}%")
        lines.append("")

    for row in benchmark_rows:
        lines += [
            f"{row.get('precision','?').upper()}",
            "-" * 88,
            f"  Estado                      : {row.get('status','')}",
        ]

        if row.get("status") == "OK":
            lines += [
                f"  VRAM pico allocated (MiB)   : {fmt_float(row.get('cuda_peak_allocated_mib'), 2)}",
                f"  VRAM pico reserved (MiB)    : {fmt_float(row.get('cuda_peak_reserved_mib'), 2)}",
                f"  Latencia media CUDA (ms)    : {fmt_float(row.get('latency_event_mean_ms'), 4)}",
                f"  Latencia mediana CUDA (ms)  : {fmt_float(row.get('latency_event_median_ms'), 4)}",
                f"  Desv. estándar CUDA (ms)    : {fmt_float(row.get('latency_event_std_ms'), 4)}",
                f"  P05 / P95 (ms)              : {fmt_float(row.get('latency_event_p05_ms'),4)} / {fmt_float(row.get('latency_event_p95_ms'),4)}",
                f"  FPS (1/media CUDA)          : {fmt_float(row.get('fps_from_event_mean'), 3)}",
                f"  Wall-clock medio (ms)       : {fmt_float(row.get('wall_mean_ms'), 4)}",
                f"  FPS wall-clock              : {fmt_float(row.get('fps_wall_clock'), 3)}",
            ]
        else:
            lines.append(
                f"  Error                       : {row.get('error')}"
            )

        lines.append("")

    lines += [
        "INTERPRETACIÓN",
        "  - Para la tabla del artículo usar siempre el MISMO input, GPU y protocolo",
        "    para Proposed y CMX baseline.",
        "  - No comparar directamente estos FPS con trabajos que incluyan pre/postprocesado",
        "    o que usen otra resolución/hardware sin explicitar la diferencia.",
        "  - Si fvcore reporta operaciones no soportadas, GFLOPs es una estimación incompleta",
        "    y debe revisarse antes de publicarse.",
        "",
        "ARCHIVOS",
        "  REPORT.txt",
        "  efficiency_summary.csv",
        "  latency_samples_fp32.csv / latency_samples_fp16.csv",
        "  model_complexity.json",
        "  environment.json",
        "  benchmark_details.json",
    ]

    (args.output / "REPORT.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
