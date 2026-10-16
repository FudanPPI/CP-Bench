# Copyright (c) Meta Platforms, Inc. and affiliates.

"""GPU SDC Detector for PyTorch operator-level testing.

Two-layer detection:

* L1 replication -- the op runs K times with deterministic settings and the
  outputs are compared bit-exactly.  This only catches NON-deterministic
  faults: a persistent fault (stuck bit, patched instruction) reproduces
  the same wrong result on every iteration and passes the comparison.
* L2 oracle -- the op is recomputed on CPU in fp64 from identical inputs
  and compared with a tolerance.  This is an independent oracle that also
  catches deterministic SDC, which replication is structurally blind to.

Both layers share one input-generation contract (see create_test_tensor
and _deterministic_rng) so the CPU oracle consumes exactly the values the
GPU run consumed.
"""

import contextlib
import logging
import random
import time
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from .ops import DTYPE_MAP, OP_REGISTRY, OpDef, PIPELINE_REGISTRY

logger = logging.getLogger("pytorch_op_test")

# Op factories draw random weights/indices/masks internally via
# torch.randn / randn_like / randint.  CPU and CUDA RNG streams yield
# different sequences for the same seed, so left alone the GPU run and the
# CPU-fp64 oracle would consume different inputs and every fault-free run
# would report SDC.  While an op executes, redirect these three functions
# to CPU generation followed by an explicit move to the target device.
_RAND_FNS = ("randn", "randn_like", "randint")


def _cpu_random_wrapper(orig_fn, follows_input_device):
    def wrapper(*args, **kwargs):
        device = kwargs.pop("device", None)
        # Generate without requires_grad and re-enable after the move:
        # .to() on a grad-requiring leaf returns a non-leaf tensor, which
        # would break ops that read .grad afterward.
        requires_grad = kwargs.pop("requires_grad", False)
        target_dtype = kwargs.get("dtype", None)
        # *_like inherits the input tensor's device/dtype when no explicit
        # ones are given; remember them so the CPU result can be moved back.
        if follows_input_device and args and isinstance(args[0], torch.Tensor):
            ref = args[0]
            if device is None:
                device = ref.device
            if target_dtype is None:
                target_dtype = ref.dtype
        # Floating draws are generated at fp32 (the precision under test)
        # and cast afterwards, so an fp64 oracle consumes the same values
        # as the fp32 GPU run instead of an independent fp64 sequence.
        gen_dtype = (torch.float32
                     if target_dtype is not None
                     and target_dtype.is_floating_point
                     else target_dtype)
        kwargs["device"] = "cpu"
        kwargs["dtype"] = gen_dtype
        t = orig_fn(*args, **kwargs)
        if device is not None or gen_dtype != target_dtype:
            t = t.to(device=device, dtype=target_dtype)
        if requires_grad:
            t.requires_grad_(True)
        return t
    return wrapper


@contextlib.contextmanager
def _deterministic_rng(seed: int = 42):
    """Scope in which random draws are value-identical across devices."""
    torch.manual_seed(seed)
    saved = {name: getattr(torch, name) for name in _RAND_FNS}
    try:
        for name in _RAND_FNS:
            setattr(torch, name,
                    _cpu_random_wrapper(saved[name], name.endswith("_like")))
        yield
    finally:
        for name, fn in saved.items():
            setattr(torch, name, fn)


# --calibrate warns when fault-free baseline error sits within this factor
# of the effective oracle tolerance (false-SSC risk on other GPUs/builds).
CALIBRATE_HEADROOM_WARN = 4.0


class GPUSDCDetector:
    """Detects Silent Data Corruption by checking op determinism on GPU."""

    # Default tensor sizes: progressive memory coverage.
    # Use --sizes for custom shapes (e.g. rectangular: 1024x16384).
    DEFAULT_SIZES = [
        (1024, 1024),
        (4096, 4096),
        (8192, 8192),
    ]

    def __init__(
        self,
        gpu_id: int = 0,
        num_iterations: int = 3,
        test_sizes: Optional[List[Tuple[int, int]]] = None,
        run_oracle: bool = True,
        oracle_rtol: float = 1e-3,
        oracle_atol: float = 1e-3,
        seed: int = 42,
        calibrate: bool = False,
    ):
        self.gpu_id = gpu_id
        self.num_iterations = num_iterations
        self.current_dtype = torch.float32
        self.device = f"cuda:{gpu_id}"
        self.run_oracle = run_oracle
        self.oracle_rtol = oracle_rtol
        self.oracle_atol = oracle_atol
        self.seed = seed
        self.calibrate = calibrate
        self._failed_operations: List[str] = []
        self._oracle_failed: List[str] = []
        self._oracle_unavailable: List[str] = []
        self._calibration_records: List[dict] = []
        self._test_sizes = test_sizes or self.DEFAULT_SIZES

    def create_test_tensor(
        self, shape: Tuple[int, ...], dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Create a test tensor with a fixed seed.

        Generated on the CPU and moved to the target device: CPU and CUDA
        RNG streams produce different sequences for the same seed, so
        generating on the device would hand the CPU oracle and the GPU run
        different inputs (fault-free runs would report SDC).  Floating
        draws always generate at fp32 and are cast afterwards, so an fp64
        oracle consumes the same values instead of a fresh fp64 sequence.
        """
        dtype = dtype or self.current_dtype
        torch.manual_seed(self.seed)
        return torch.randn(shape, dtype=torch.float32, device="cpu").to(
            dtype=dtype, device=self.device
        )

    def _resolve_dtypes(self, dtypes: Optional[List[str]]) -> List[torch.dtype]:
        """Validate and resolve dtype strings to torch.dtype objects."""
        if dtypes is None:
            dtypes = ["float32"]
        resolved = []
        for dt_name in dtypes:
            dt_name = dt_name.strip()
            if dt_name not in DTYPE_MAP:
                raise ValueError(
                    f"Unknown dtype: {dt_name}. "
                    f"Available: {', '.join(DTYPE_MAP.keys())}"
                )
            resolved.append(DTYPE_MAP[dt_name])
        return resolved

    def _check_gpu(self):
        """Validate that the target GPU is available."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if self.gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"GPU {self.gpu_id} not found. Available: {torch.cuda.device_count()}"
            )

    def _run_op_tests(
        self, categories: Optional[List[str]], resolved_dtypes: List[torch.dtype]
    ) -> Tuple[int, int]:
        """Run all registered op tests across dtypes and sizes. Returns (total, passed)."""
        total_ops = 0
        passed_ops = 0
        for dtype in resolved_dtypes:
            self.current_dtype = dtype
            if len(resolved_dtypes) > 1:
                logger.info(f"Testing dtype: {dtype}")
            for size in self._test_sizes:
                logger.info(f"Testing with tensor size: {size}")
                for category, ops in self._get_ops_by_category(categories):
                    cat_label = category.replace("_", " ").title()
                    logger.info(f"--- {cat_label} Operations ---")
                    for op_def in ops:
                        operation = op_def.factory(self, size)
                        total_ops += 1
                        if self._test_operation(
                            op_def.name,
                            operation,
                            oracle_factory=op_def.factory,
                            size=size,
                        ):
                            passed_ops += 1
        return total_ops, passed_ops

    def _run_pipeline_tests(self) -> Tuple[int, int]:
        """Run composite pipeline tests. Returns (total, passed)."""
        total_ops = 0
        passed_ops = 0
        logger.info("--- Composite Pipeline Tests ---")
        size = (1024, 1024)
        for pipeline_def in PIPELINE_REGISTRY:
            operation = pipeline_def.factory(self, size)
            total_ops += 1
            if self._test_operation(
                f"[Pipeline] {pipeline_def.name}",
                operation,
                oracle_factory=pipeline_def.factory,
                size=size,
            ):
                passed_ops += 1
        return total_ops, passed_ops

    def _log_results(self, total_ops: int, elapsed: float):
        """Log final pass/fail summary."""
        n_nondet = len(self._failed_operations)
        n_det = len(self._oracle_failed)
        if n_nondet or n_det:
            logger.error(
                f"FAILED: {n_nondet} non-deterministic operation(s), "
                f"{n_det} deterministic SDC(s) on GPU {self.gpu_id}"
            )
            for i, op in enumerate(self._failed_operations, 1):
                logger.error(f"  [non-det {i}] {op}")
            for i, op in enumerate(self._oracle_failed, 1):
                logger.error(f"  [det-sdc {i}] {op}")
            if self._oracle_unavailable:
                logger.warning(
                    f"  oracle unavailable for {len(self._oracle_unavailable)} "
                    f"operation(s): {', '.join(self._oracle_unavailable)}"
                )
        else:
            logger.info(
                f"ALL TESTS PASSED - GPU {self.gpu_id} deterministic and "
                f"oracle-clean across {total_ops} operations in {elapsed:.1f}s"
            )

    def run_all_tests(
        self,
        categories: Optional[List[str]] = None,
        dtypes: Optional[List[str]] = None,
        run_pipelines: bool = False,
    ) -> dict:
        """Run all registered op tests and return results dict.

        Returns:
            dict with keys: ops_total, ops_passed, ops_failed,
            elapsed_seconds, failed_operations
        """
        resolved_dtypes = self._resolve_dtypes(dtypes)
        self._check_gpu()
        self._enable_deterministic_mode()

        logger.info(
            f"Testing GPU {self.gpu_id}: {torch.cuda.get_device_name(self.gpu_id)}"
        )
        logger.info(f"Iterations per operation: {self.num_iterations}")
        if self.calibrate:
            logger.info(
                "CALIBRATION MODE: one fault-free pass per op measuring "
                "CPU-fp64 vs GPU-fp32 baseline error; oracle tolerances are "
                "reported, never auto-adjusted"
            )

        start_time = time.time()
        self._failed_operations = []
        self._oracle_failed = []
        self._oracle_unavailable = []
        self._calibration_records = []

        total_ops, passed_ops = self._run_op_tests(categories, resolved_dtypes)

        if run_pipelines and PIPELINE_REGISTRY:
            self.current_dtype = resolved_dtypes[0]
            pipe_total, pipe_passed = self._run_pipeline_tests()
            total_ops += pipe_total
            passed_ops += pipe_passed

        elapsed = time.time() - start_time
        if self.calibrate:
            self._log_calibration_summary()
        else:
            self._log_results(total_ops, elapsed)

        return {
            "ops_total": total_ops,
            "ops_passed": passed_ops,
            "ops_failed": len(self._failed_operations),
            "failed_operations": list(self._failed_operations),
            "deterministic_sdc": len(self._oracle_failed),
            "oracle_failed_operations": list(self._oracle_failed),
            "oracle_unavailable": len(self._oracle_unavailable),
            "oracle_unavailable_operations": list(self._oracle_unavailable),
            "seed": self.seed,
            "calibrate": self.calibrate,
            "calibration": list(self._calibration_records),
            "elapsed_seconds": round(elapsed, 2),
        }

    def _log_calibration_summary(self):
        """Log per-op baseline headroom measured by --calibrate.

        Headroom = effective tolerance / fault-free baseline max abs diff.
        Ops below CALIBRATE_HEADROOM_WARN risk false SDC verdicts on other
        GPUs/torch builds; the operator should raise --oracle-rtol/--oracle-atol.
        """
        numeric = [
            r for r in self._calibration_records
            if r.get("status") in ("within", "exceeds")
        ]
        n_exceeds = sum(1 for r in numeric if r["status"] == "exceeds")
        n_within = len(numeric) - n_exceeds
        n_unavailable = sum(
            1 for r in self._calibration_records if r.get("status") == "unavailable"
        )
        headrooms = [r["headroom"] for r in numeric if r.get("headroom")]
        tight = [
            r for r in numeric
            if r.get("headroom") is not None
            and r["status"] == "within"
            and r["headroom"] < CALIBRATE_HEADROOM_WARN
        ]
        logger.info(
            f"CALIBRATION SUMMARY: {len(numeric)} op(s) measured "
            f"({n_within} within tolerance, {n_exceeds} exceed"
            f"{f', {n_unavailable} oracle-unavailable' if n_unavailable else ''})"
        )
        if headrooms:
            logger.info(
                f"  min headroom {min(headrooms):.1f}x "
                f"(warn threshold {CALIBRATE_HEADROOM_WARN:.0f}x)"
            )
        for r in [r for r in numeric if r["status"] == "exceeds"]:
            logger.warning(
                f"  [exceeds] {r['op_name']} ({r['dtype']} {r['size']}): "
                f"baseline {r['baseline_max_diff']:.3g} > tol {r['tol']:.3g}"
            )
        for r in tight:
            logger.warning(
                f"  [tight]   {r['op_name']} ({r['dtype']} {r['size']}): "
                f"headroom {r['headroom']:.1f}x"
            )
        if n_exceeds or tight:
            logger.warning(
                "  raise --oracle-rtol/--oracle-atol before trusting SDC "
                "verdicts; calibration never loosens tolerance automatically"
            )

    def _enable_deterministic_mode(self):
        """Enable deterministic settings for reproducible results."""
        seed = self.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TF32 truncates fp32 inputs to a 10-bit mantissa on sm_80+; left
        # on, it pushes GPU results ~1e-1 away from the fp64 CPU oracle and
        # trips the tolerance check on fault-free runs.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def _run_operation_on_gpu(self, operation: Callable) -> torch.Tensor:
        """Run an operation on the GPU and return result on CPU."""
        with torch.cuda.device(self.gpu_id):
            result = operation()
            if isinstance(result, torch.Tensor):
                return result.cpu()
            return result

    def _test_operation(
        self,
        op_name: str,
        operation: Callable,
        oracle_factory: Optional[Callable] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Test an operation: L1 replication + optional L2 CPU-fp64 oracle."""
        if self.calibrate:
            return self._calibrate_operation(
                op_name, operation, oracle_factory, size
            )
        try:
            outputs = []
            with _deterministic_rng(self.seed):
                for _ in range(self.num_iterations):
                    self._enable_deterministic_mode()
                    output = self._run_operation_on_gpu(operation)
                    outputs.append(output)

            replication_ok = self._compare_outputs(outputs, op_name)
        except Exception as e:
            logger.error(f"EXCEPTION in {op_name}: {str(e)}")
            self._failed_operations.append(op_name)
            return False

        if self.run_oracle and oracle_factory is not None and replication_ok:
            if not self._run_oracle_check(op_name, oracle_factory, size,
                                          outputs[0]):
                return False
        return replication_ok

    def _run_oracle_check(
        self,
        op_name: str,
        factory: Callable,
        size: Tuple[int, int],
        gpu_output: torch.Tensor,
    ) -> bool:
        """L2 oracle: recompute the op on CPU in fp64 from identical inputs.

        Replication (L1) is structurally blind to deterministic SDC: a
        persistent fault produces the same wrong result on every iteration
        and the bit-exact comparison confirms the agreement.  The CPU-fp64
        recomputation is an independent oracle.  Ops whose factories cannot
        run on CPU are reported as oracle-unavailable, not as failures.
        """
        result = self._oracle_compare(op_name, factory, size, gpu_output)
        if result is None:
            return True
        if result["sdc"]:
            self._oracle_failed.append(op_name)
            return False
        return True

    def _oracle_compare(
        self,
        op_name: str,
        factory: Callable,
        size: Tuple[int, int],
        gpu_output: torch.Tensor,
        log_verdict: bool = True,
    ) -> Optional[dict]:
        """Compare one GPU output against the CPU-fp64 oracle.

        Returns None when the op has no CPU oracle (unavailable); otherwise
        a dict: sdc, reason, baseline_max_diff, tol, mismatches, numel.
        ``log_verdict=False`` suppresses SDC/PASS verdict logs, used by
        --calibrate where the same comparison is a baseline measurement.
        """
        shadow = GPUSDCDetector(
            gpu_id=self.gpu_id,
            num_iterations=1,
            test_sizes=[size],
            run_oracle=False,
            seed=self.seed,
        )
        shadow.device = "cpu"
        shadow.current_dtype = torch.float64
        try:
            # No torch.no_grad() here: backward_pass ops legitimately call
            # .backward() inside the factory, which cannot build a graph
            # under no_grad.
            with _deterministic_rng(self.seed):
                operation = factory(shadow, size)
                golden = operation()
            golden = golden.detach().to(torch.float64).cpu()
        except Exception as e:
            logger.warning(
                f"ORACLE UNAVAILABLE in {op_name}: {type(e).__name__}: {e}"
            )
            self._oracle_unavailable.append(op_name)
            return None

        out = gpu_output.detach().to(torch.float64).cpu()
        if out.shape != golden.shape:
            if log_verdict:
                logger.error(
                    f"DETERMINISTIC SDC (oracle) in {op_name}: shape mismatch "
                    f"{tuple(out.shape)} vs {tuple(golden.shape)}"
                )
            return {
                "sdc": True, "reason": "shape mismatch",
                "baseline_max_diff": None, "tol": None,
                "mismatches": None, "numel": None,
            }
        if out.numel() == 0:
            if log_verdict:
                logger.info(f"ORACLE PASS {op_name} (empty output)")
            return {
                "sdc": False, "reason": "empty",
                "baseline_max_diff": 0.0, "tol": self.oracle_atol,
                "mismatches": 0, "numel": 0,
            }
        nan_out, nan_gold = torch.isnan(out), torch.isnan(golden)
        inf_out, inf_gold = torch.isinf(out), torch.isinf(golden)
        if not torch.equal(nan_out, nan_gold) or not torch.equal(inf_out, inf_gold):
            if log_verdict:
                logger.error(
                    f"DETERMINISTIC SDC (oracle) in {op_name}: "
                    f"NaN/Inf pattern differs from the fp64 reference"
                )
            return {
                "sdc": True, "reason": "nan_inf pattern",
                "baseline_max_diff": None, "tol": None,
                "mismatches": None, "numel": None,
            }
        if inf_out.any() and not torch.equal(out[inf_out], golden[inf_gold]):
            if log_verdict:
                logger.error(
                    f"DETERMINISTIC SDC (oracle) in {op_name}: "
                    f"Inf sign differs from the fp64 reference"
                )
            return {
                "sdc": True, "reason": "inf sign",
                "baseline_max_diff": None, "tol": None,
                "mismatches": None, "numel": None,
            }
        finite = ~nan_out & ~inf_out
        max_diff, mismatches = 0.0, 0
        tol = self.oracle_atol
        if finite.any():
            diff = (out[finite] - golden[finite]).abs()
            max_diff = diff.max().item()
            tol = (self.oracle_atol
                   + self.oracle_rtol * golden[finite].abs().max().item())
            mismatches = int((diff > tol).sum().item())
        sdc = max_diff > tol
        if log_verdict:
            if sdc:
                logger.error(
                    f"DETERMINISTIC SDC (oracle) in {op_name}: "
                    f"max abs diff {max_diff:.6g} > tolerance {tol:.6g}, "
                    f"mismatched={mismatches}/{int(finite.sum().item())}"
                )
            else:
                logger.info(f"ORACLE PASS {op_name} (max abs diff {max_diff:.6g})")
        return {
            "sdc": sdc, "reason": None,
            "baseline_max_diff": max_diff, "tol": tol,
            "mismatches": mismatches, "numel": int(finite.sum().item()),
        }

    def _calibrate_operation(
        self,
        op_name: str,
        operation: Callable,
        oracle_factory: Optional[Callable],
        size: Optional[Tuple[int, int]],
    ) -> bool:
        """--calibrate: measure one fault-free run's baseline oracle error.

        Never adjusts tolerance; records per-op baseline/headroom for the
        report so operators can size --oracle-rtol/--oracle-atol before a
        detection run.  Always returns True: calibration is not a verdict.
        """
        dtype_label = str(self.current_dtype).split(".")[-1]
        record = {
            "op_name": op_name,
            "dtype": dtype_label,
            "size": list(size) if size is not None else None,
        }
        try:
            with _deterministic_rng(self.seed):
                self._enable_deterministic_mode()
                gpu_output = self._run_operation_on_gpu(operation)
        except Exception as e:
            logger.warning(f"CALIBRATE SKIP {op_name}: {type(e).__name__}: {e}")
            record.update(status="error", reason=str(e))
            self._calibration_records.append(record)
            return True

        if oracle_factory is None:
            logger.info(f"CALIBRATE {op_name}: no oracle available")
            record.update(status="no_oracle")
            self._calibration_records.append(record)
            return True

        result = self._oracle_compare(
            op_name, oracle_factory, size, gpu_output, log_verdict=False
        )
        if result is None:
            record.update(status="unavailable")
            self._calibration_records.append(record)
            return True

        baseline = result["baseline_max_diff"]
        tol = result["tol"]
        record.update(
            reason=result["reason"],
            baseline_max_diff=baseline,
            tol=tol,
            mismatches=result["mismatches"],
            numel=result["numel"],
        )
        if result["sdc"]:
            record.update(status="exceeds", headroom=None)
            logger.warning(
                f"CALIBRATE {op_name} ({dtype_label} {size}): baseline "
                f"{baseline:.3g} EXCEEDS tol {tol:.3g} "
                f"(mismatched {result['mismatches']}/{result['numel']})"
            )
        else:
            headroom = (tol / baseline) if baseline and baseline > 0 else None
            record.update(status="within", headroom=headroom)
            if headroom is not None and headroom < CALIBRATE_HEADROOM_WARN:
                logger.warning(
                    f"CALIBRATE {op_name} ({dtype_label} {size}): baseline "
                    f"{baseline:.3g}, tol {tol:.3g}, headroom {headroom:.1f}x "
                    f"TIGHT (<{CALIBRATE_HEADROOM_WARN:.0f}x)"
                )
            else:
                hr_text = f"headroom {headroom:.1f}x" if headroom else "zero baseline"
                logger.info(
                    f"CALIBRATE {op_name} ({dtype_label} {size}): baseline "
                    f"{baseline:.3g}, tol {tol:.3g}, {hr_text}"
                )
        self._calibration_records.append(record)
        return True

    def _compare_outputs(self, outputs: List[torch.Tensor], op_name: str) -> bool:
        """Compare multiple outputs from the same operation."""
        reference = outputs[0]

        for _i, output in enumerate(outputs[1:], start=1):
            if output.shape != reference.shape:
                logger.error(f"SHAPE MISMATCH in {op_name}")
                self._failed_operations.append(op_name)
                return False

            # Bit-exact comparison — any difference at all is SDC.
            # Use NaN-aware comparison: IEEE 754 says NaN != NaN, so
            # torch.equal returns False when both tensors have NaN in
            # the same positions.  Treat matching NaN positions as equal.
            nan_ref = torch.isnan(reference)
            nan_out = torch.isnan(output)
            has_nan = nan_ref.any() or nan_out.any()
            if has_nan:
                nan_positions_match = torch.equal(nan_ref, nan_out)
                non_nan = ~nan_ref
                non_nan_equal = not non_nan.any() or torch.equal(
                    reference[non_nan], output[non_nan]
                )
                tensors_equal = nan_positions_match and non_nan_equal
            else:
                tensors_equal = torch.equal(reference, output)

            if not tensors_equal:
                # Upcast to float32 only for the detailed mismatch report
                ref_f32 = reference.float()
                out_f32 = output.float()
                diff = torch.abs(ref_f32 - out_f32)
                # Mask NaN positions so they don't poison stats
                nan_mask = torch.isnan(diff)
                if nan_mask.any():
                    diff = diff.clone()
                    diff[nan_mask] = 0.0
                mismatches = int((diff > 0).sum().item())
                total = reference.numel()
                max_diff = diff.max().item()
                logger.error(
                    f"CORRUPTION DETECTED in {op_name}: "
                    f"max_diff={max_diff:.2e}, "
                    f"mismatched={mismatches}/{total} "
                    f"({100 * mismatches / total:.2f}%)"
                )
                self._print_detailed_mismatch(ref_f32, out_f32, diff, op_name)
                self._failed_operations.append(op_name)
                return False

        logger.info(
            f"PASSED {op_name} (all {self.num_iterations} iterations identical)"
        )
        return True

    def _print_detailed_mismatch(
        self,
        reference: torch.Tensor,
        output: torch.Tensor,
        diff: torch.Tensor,
        op_name: str,
    ):
        """Print detailed information about the mismatch."""
        max_diff_idx = diff.argmax()
        max_diff_pos = np.unravel_index(max_diff_idx.item(), diff.shape)

        logger.error(f"  Location of max difference: {max_diff_pos}")
        logger.error(f"  Expected: {reference.flatten()[max_diff_idx].item():.10e}")
        logger.error(f"  Actual:   {output.flatten()[max_diff_idx].item():.10e}")

        num_samples = min(10, diff.numel())
        top_diffs = torch.topk(diff.flatten(), num_samples)
        logger.error(f"  Top {num_samples} mismatched values:")
        for idx in top_diffs.indices:
            pos = np.unravel_index(idx.item(), diff.shape)
            logger.error(
                f"    {pos}: "
                f"expected={reference.flatten()[idx].item():.10e}, "
                f"actual={output.flatten()[idx].item():.10e}, "
                f"diff={diff.flatten()[idx].item():.10e}"
            )

        logger.error(
            f"  Reference: mean={reference.mean().item():.6e}, "
            f"std={reference.std().item():.6e}"
        )
        logger.error(
            f"  Corrupted: mean={output.mean().item():.6e}, "
            f"std={output.std().item():.6e}"
        )

    def _get_ops_by_category(
        self, categories: Optional[List[str]] = None
    ) -> List[Tuple[str, List[OpDef]]]:
        """Return ops grouped by category, preserving registration order."""
        grouped: OrderedDict[str, List[OpDef]] = OrderedDict()
        for op_def in OP_REGISTRY:
            if categories and op_def.category not in categories:
                continue
            grouped.setdefault(op_def.category, []).append(op_def)
        return list(grouped.items())

    def list_ops(self):
        """Print all registered ops grouped by category."""
        grouped: OrderedDict[str, List[str]] = OrderedDict()
        for op_def in OP_REGISTRY:
            grouped.setdefault(op_def.category, []).append(op_def.name)

        print("Registered operations:\n")
        total = 0
        for category, names in grouped.items():
            cat_label = category.replace("_", " ").title()
            print(f"  {cat_label} ({len(names)}):")
            for name in names:
                print(f"    - {name}")
            total += len(names)
            print()

        print(f"Total: {total} operations across {len(grouped)} categories")

        if PIPELINE_REGISTRY:
            print(f"\nComposite pipelines ({len(PIPELINE_REGISTRY)}):")
            for p in PIPELINE_REGISTRY:
                print(f"    - {p.name}")

        print(f"\nAvailable categories: {', '.join(grouped.keys())}")
        print(f"Available dtypes: {', '.join(DTYPE_MAP.keys())}")
