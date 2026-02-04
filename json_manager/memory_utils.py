"""Utility functions for managing GPU memory in JSON Manager GUI."""

import logging
import tkinter as tk
from tkinter import ttk, messagebox
from typing import TYPE_CHECKING
import gc
import torch
from comfy.model_management import get_free_memory, unload_all_models, soft_empty_cache, current_loaded_models


if TYPE_CHECKING:
    from json_gui.json_manager.json_manager_gui import JSONManagerApp


def check_memory_available(required_gb: float = 10.5) -> bool:
    """Check if enough contiguous memory is available.

    Args:
        required_gb: Required memory in GB (default 10.5 for SD3.5)

    Returns:
        True if memory is available, False if too fragmented
    """
    if not torch.cuda.is_available():
        return True

    try:
        required_bytes = int(required_gb * 1024 * 1024 * 1024)
        free_memory = get_free_memory() + 1024 * 1024 * 512  # Add 512MB buffer
        return free_memory >= required_bytes
    except Exception as e:
        logging.warning("Failed to check memory: %s", e)
        return True  # Assume OK if check fails


def manual_cleanup(inst: "JSONManagerApp") -> None:
    """Manually clean up GPU memory."""
    inst.status_var.set("Cleaning memory...")
    inst.root.update()

    cleanup_vram()

    # Reset warning flag after manual cleanup
    inst.memory_warning_shown = False

    # Show current memory status
    if torch.cuda.is_available():
        try:
            free_cuda, total_cuda = torch.cuda.mem_get_info()
            free_gb = free_cuda / (1024**3)
            total_gb = total_cuda / (1024**3)
            inst.status_var.set(f"Memory cleaned. Free: {free_gb:.1f}GB / {total_gb:.1f}GB")
            messagebox.showinfo(
                "Memory Cleanup",
                f"Memory cleanup complete.\n\n"
                f"Free GPU Memory: {free_gb:.1f} GB\n"
                f"Total GPU Memory: {total_gb:.1f} GB",
            )
        except Exception:
            inst.status_var.set("Memory cleaned.")
    else:
        inst.status_var.set("Memory cleaned (no GPU).")


def cleanup_vram() -> None:
    """Clean up VRAM using ComfyUI's memory management."""
    try:
        # Use ComfyUI's unload to properly release models
        unload_all_models()
        soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        logging.warning("Failed to cleanup VRAM: %s", e)


def check_memory_fragmentation(inst: "JSONManagerApp") -> None:
    """Check memory fragmentation after execution and warn user if needed."""
    if not torch.cuda.is_available():
        return

    try:
        # Get memory stats
        stats = torch.cuda.memory_stats()
        mem_reserved = stats.get("reserved_bytes.all.current", 0)
        mem_active = stats.get("active_bytes.all.current", 0)
        free_cuda, _total_cuda = torch.cuda.mem_get_info()

        # Calculate fragmentation: memory reserved but not active
        fragmented = mem_reserved - mem_active
        fragmented_gb = fragmented / (1024**3)
        free_gb = free_cuda / (1024**3)

        logging.info(
            "Memory status: Free=%.2fGB, Reserved=%.2fGB, Active=%.2fGB, Fragmented=%.2fGB",
            free_gb,
            mem_reserved / (1024**3),
            mem_active / (1024**3),
            fragmented_gb,
        )

        # Warn if free memory is low and there's significant fragmentation
        # Your model needs ~10.5GB, so warn if we have less than 12GB free
        if free_gb < 12.0 and not inst.memory_warning_shown:
            inst.memory_warning_shown = True
            inst.root.after(
                0,
                lambda: messagebox.showwarning(
                    "Low Memory Warning",
                    f"GPU memory is getting low ({free_gb:.1f}GB free).\n\n"
                    "If the next execution fails, please restart the application.",
                ),
            )

    except Exception as e:
        logging.warning("Failed to check memory fragmentation: %s", e)


def show_memory_details(inst: "JSONManagerApp") -> None:
    """Show detailed GPU memory information in a popup window."""
    if not torch.cuda.is_available():
        messagebox.showinfo("Memory Info", "No CUDA GPU available.")
        return

    try:
        # Gather all memory information
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        device_props = torch.cuda.get_device_properties(device)
        # Basic memory info
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        used_cuda = total_cuda - free_cuda
        # PyTorch memory stats
        stats = torch.cuda.memory_stats(device)
        mem_allocated = stats.get("allocated_bytes.all.current", 0)
        mem_reserved = stats.get("reserved_bytes.all.current", 0)
        mem_active = stats.get("active_bytes.all.current", 0)
        mem_inactive = mem_reserved - mem_active
        # Peak memory
        mem_allocated_peak = stats.get("allocated_bytes.all.peak", 0)
        mem_reserved_peak = stats.get("reserved_bytes.all.peak", 0)
        # Allocation counts
        num_allocs = stats.get("allocation.all.current", 0)
        num_allocs_peak = stats.get("allocation.all.peak", 0)
        # Segment info (fragmentation indicator)
        num_segments = stats.get("segment.all.current", 0)
        num_segments_peak = stats.get("segment.all.peak", 0)
        large_pool_allocated = stats.get("allocated_bytes.large_pool.current", 0)
        small_pool_allocated = stats.get("allocated_bytes.small_pool.current", 0)
        # OOM stats
        num_ooms = stats.get("num_ooms", 0)
        num_alloc_retries = stats.get("num_alloc_retries", 0)

        # Calculate fragmentation metrics
        fragmented = mem_reserved - mem_active
        fragmentation_pct = (fragmented / mem_reserved * 100) if mem_reserved > 0 else 0

        # ComfyUI loaded models
        loaded_models_info = ""
        try:
            loaded_models = current_loaded_models
            if loaded_models:
                loaded_models_info = f"\n{'=' * 50}\n"
                loaded_models_info += "COMFYUI LOADED MODELS:\n"
                loaded_models_info += f"{'=' * 50}\n"
                for i, lm in enumerate(loaded_models):
                    model_name = (
                        lm.model.model.__class__.__name__ if hasattr(lm.model, "model") else str(type(lm.model))
                    )
                    model_mem = lm.model_memory() / (1024**3)
                    loaded_models_info += f"  [{i + 1}] {model_name}: {model_mem:.2f} GB\n"
        except Exception as e:
            loaded_models_info = f"\n(Could not get loaded models: {e})\n"

        def to_gb(b: int) -> str:
            return f"{b / (1024**3):.3f} GB"

        def to_mb(b: int) -> str:
            return f"{b / (1024**2):.1f} MB"

        # Build detailed report
        report = f"""{'=' * 50}
GPU MEMORY REPORT
{'=' * 50}

DEVICE INFO:
Name: {device_name}
Total Memory: {to_gb(device_props.total_memory)}
Compute Capability: {device_props.major}.{device_props.minor}
Multi Processors: {device_props.multi_processor_count}

{'=' * 50}
CUDA MEMORY (Hardware Level):
{'=' * 50}
Total:     {to_gb(total_cuda)}
Used:      {to_gb(used_cuda)} ({used_cuda / total_cuda * 100:.1f}%)
Free:      {to_gb(free_cuda)} ({free_cuda / total_cuda * 100:.1f}%)

{'=' * 50}
PYTORCH MEMORY (Software Level):
{'=' * 50}
Allocated (current): {to_gb(mem_allocated)}
Allocated (peak):    {to_gb(mem_allocated_peak)}
Reserved (current):  {to_gb(mem_reserved)}
Reserved (peak):     {to_gb(mem_reserved_peak)}

{'=' * 50}
FRAGMENTATION ANALYSIS:
{'=' * 50}
Active Memory:       {to_gb(mem_active)}
Inactive (cached):   {to_gb(mem_inactive)}
Fragmented:          {to_gb(fragmented)}
Fragmentation:       {fragmentation_pct:.1f}%

Memory Segments:     {num_segments} (peak: {num_segments_peak})
Active Allocations:  {num_allocs} (peak: {num_allocs_peak})

Large Pool:          {to_gb(large_pool_allocated)}
Small Pool:          {to_mb(small_pool_allocated)}

{'=' * 50}
HEALTH INDICATORS:
{'=' * 50}
Out of Memory Events: {num_ooms}
Allocation Retries:   {num_alloc_retries}

Status: {'⚠️ HIGH FRAGMENTATION' if fragmentation_pct > 30 else '✅ OK' if fragmentation_pct < 15 else '⚡ MODERATE'}

Your model needs ~10.5 GB
Available free:  {to_gb(free_cuda)}
Can load model:  {'✅ YES' if free_cuda > 11 * 1024**3 else '❌ NO - Restart recommended'}
{loaded_models_info}
{'=' * 50}
"""

        # Create a new window with scrollable text
        detail_window = tk.Toplevel(inst.root)
        detail_window.title("GPU Memory Details")
        detail_window.geometry("650x700")
        detail_window.transient(inst.root)

        # Text widget with scrollbar
        text_frame = ttk.Frame(detail_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(
            text_frame,
            wrap="none",
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set,
        )
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert("1.0", report)
        text_widget.config(state="disabled")  # Read-only

        # Buttons at bottom
        btn_frame = ttk.Frame(detail_window)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(
            btn_frame,
            text="Refresh",
            command=lambda: _refresh_memory_window(text_widget),
        ).pack(side="left", padx=5)

        def clean_action() -> None:
            cleanup_vram()
            _refresh_memory_window(text_widget)

        ttk.Button(
            btn_frame,
            text="Clean Memory",
            command=clean_action,
        ).pack(side="left", padx=5)

        ttk.Button(btn_frame, text="Close", command=detail_window.destroy).pack(side="right", padx=5)

    except Exception as e:
        logging.exception("Failed to get memory details")
        messagebox.showerror("Error", f"Failed to get memory details:\n{e}")


def _refresh_memory_window(text_widget: tk.Text) -> None:
    """Refresh the memory details in an existing window."""
    text_widget.config(state="normal")
    text_widget.delete("1.0", "end")

    # Regenerate report (simplified version for refresh)
    try:
        device = torch.cuda.current_device()
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats(device)
        mem_reserved = stats.get("reserved_bytes.all.current", 0)
        mem_active = stats.get("active_bytes.all.current", 0)
        fragmented = mem_reserved - mem_active
        fragmentation_pct = (fragmented / mem_reserved * 100) if mem_reserved > 0 else 0

        report = f"""QUICK REFRESH - {torch.cuda.get_device_name(device)}

Free: {free_cuda / (1024**3):.2f} GB / {total_cuda / (1024**3):.2f} GB
Reserved: {mem_reserved / (1024**3):.2f} GB
Active: {mem_active / (1024**3):.2f} GB
Fragmentation: {fragmentation_pct:.1f}%

Status: {'⚠️ HIGH FRAGMENTATION' if fragmentation_pct > 30 else '✅ OK' if fragmentation_pct < 15 else '⚡ MODERATE'}
Can load 10.5GB model: {'✅ YES' if free_cuda > 11 * 1024**3 else '❌ NO'}
"""
        text_widget.insert("1.0", report)
    except Exception as e:
        text_widget.insert("1.0", f"Error: {e}")

    text_widget.config(state="disabled")
