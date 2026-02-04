"""Module for displaying a loading modal with redirected logging output in a Tkinter GUI."""

import sys
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Callable

__all__ = ["show_loading_modal"]


class TpLevProWin(tk.Toplevel):
    """Toplevel window with a progress window attribute."""

    @property
    def progress_window(self) -> tk.Toplevel:
        """Get the progress window."""
        return self._progress_window

    def __init__(self, parent: tk.Tk, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._progress_window: tk.Toplevel = self._show_progress_window(parent)

    @staticmethod
    def _show_progress_window(parent: tk.Tk) -> tk.Toplevel:
        """Show a progress window with logging output redirected to it."""
        win = tk.Toplevel(parent)
        win.title("Progreso")

        width = win.winfo_screenwidth()
        height = 400

        win.geometry(f"{width}x{height}")

        text = tk.Text(win, height=15, width=80, state="disabled")
        text_scrollbar = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=text_scrollbar.set)
        text_scrollbar.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)

        writer = _TkTextWriter(text)

        # ---- logging ----
        handler = _TkTextHandler(writer)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        # ---- stdout / stderr ----
        sys.stdout = _TextRedirector(writer, sys.__stdout__)
        sys.stderr = _TextRedirector(writer, sys.__stderr__)

        return win


class _TkTextWriter:
    """Writer that outputs to a Tkinter text widget."""

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write_to_widget(self, msg) -> None:
        """Write message to the text widget in a thread-safe manner."""
        if not msg:
            return

        def append() -> None:
            """
            Append message to text widget.
            It handles both normal messages and tqdm-style progress updates.
            """

            if not self.text_widget.winfo_exists():
                return

            self.text_widget.configure(state="normal")
            if "\r" in msg:
                chunks = msg.split("\r")
                self.text_widget.insert(tk.END, chunks[0])
                for chunk in chunks[1:]:
                    self.text_widget.delete("end-1c linestart", "end-1c")
                    self.text_widget.insert(tk.END, chunk)
            else:
                self.text_widget.insert(tk.END, msg)
            self.text_widget.see(tk.END)
            self.text_widget.configure(state="disabled")

        try:
            self.text_widget.after(0, append)
        except tk.TclError:
            pass


class _TkTextHandler(logging.Handler):
    """Logging handler that writes to TkTextWriter."""

    def __init__(self, writer):
        super().__init__()
        self.writer = writer

    def emit(self, record) -> None:
        """Emit a log record to the TkTextWriter."""
        # Check if this is a progress bar update from child process
        is_progress = getattr(record, "is_progress", False)
        if is_progress:
            # For progress bars: use \r prefix (no level prefix, no newline)
            msg = "\r" + record.getMessage()
        else:
            msg = self.format(record) + "\n"
        self.writer.write_to_widget(msg)


class _TextRedirector:
    """Redirect stdout/stderr to TkTextWriter and original streams."""

    def __init__(self, writer, *orig_streams):
        self.writer = writer
        self.orig_streams = orig_streams

    def write(self, s) -> None:
        """Write to original streams and TkTextWriter."""
        for stream in self.orig_streams:
            stream.write(s)
        self.writer.write_to_widget(s)

    def flush(self) -> None:
        """Flush original streams."""
        for stream in self.orig_streams:
            stream.flush()

    def __getattr__(self, name) -> Any:
        """Delegate attribute access to the first original stream."""
        return getattr(self.orig_streams[0], name)


def _call_wrapper(parent: tk.Tk, fun: Callable, loading_win: TpLevProWin, wait: bool) -> Callable:
    """Wrapper to call a function with arguments."""

    def cleanup() -> None:
        """Cleanup function to close loading and progress windows."""
        if loading_win:
            progress_win = loading_win.progress_window
            try:
                if loading_win.winfo_exists():
                    loading_win.destroy()
            except Exception:
                pass

            if progress_win:
                _close_progress_dialog(progress_win, wait)

    def inner(*args, **kwargs) -> None:
        """Inner function to call the wrapped function."""
        try:
            logging.debug("Starting loading modal call wrapper for %s on %s", fun.__name__, parent)
            fun(*args, **kwargs)
            logging.debug("Finished loading modal call wrapper for %s on %s", fun.__name__, parent)
        except Exception as e:
            logging.exception("Error in loading modal call wrapper %s on %s", e, parent)
            error_val = e

            def show_error() -> None:
                cleanup()
                messagebox.showerror(f"Execution Error on {parent}", f"An error occurred: {error_val}")

            parent.after(0, show_error)
        else:
            parent.after(0, cleanup)

    return inner


def _close_progress_dialog(progress_win: tk.Toplevel, wait: bool) -> None:
    """Close the progress window with a confirmation dialog."""
    if not progress_win or not progress_win.winfo_exists():
        return

    try:
        progress_win.deiconify()
        progress_win.lift()
        progress_win.focus_force()

        if not wait:
            if progress_win.winfo_exists():
                progress_win.destroy()
            return

        dialog = tk.Toplevel(progress_win)
        dialog.title("Close Progress Window")
        dialog.geometry("350x120")
        dialog.transient(progress_win)
        dialog.grab_set()

        # Center dialog
        try:
            x = progress_win.winfo_x() + (progress_win.winfo_width() // 2) - 175
            y = progress_win.winfo_y() + (progress_win.winfo_height() // 2) - 60
            dialog.geometry(f"+{x}+{y}")
        except Exception:
            pass

        lbl = ttk.Label(dialog, text="Do you want to close the progress window?", padding=20)
        lbl.pack()

        def on_accept() -> None:
            try:
                if progress_win.winfo_exists():
                    progress_win.destroy()
                if dialog.winfo_exists():
                    dialog.destroy()
            except Exception:
                pass

        def on_cancel() -> None:
            try:
                if dialog.winfo_exists():
                    dialog.destroy()
            except Exception:
                pass

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill="x", pady=10)

        ttk.Button(btn_frame, text="Accept", command=on_accept).pack(side="left", expand=True, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left", expand=True, padx=5)

        # Auto-close in 5 seconds
        dialog.after(5000, on_accept)

        # Handle window close button (X)
        dialog.protocol("WM_DELETE_WINDOW", on_accept)

    except Exception as e:
        logging.exception("Error in close progress dialog: %s", e)


def _create_loading_modal(parent: tk.Tk, message="Loading...") -> TpLevProWin:
    """Show a modal loading window."""
    try:
        loading_win = TpLevProWin(parent)
        loading_win.title("")
        loading_win.geometry("250x80")
        loading_win.transient(parent)
        loading_win.grab_set()  # Hace la ventana modal
        loading_win.resizable(False, False)
        loading_win.protocol("WM_DELETE_WINDOW", lambda: None)  # Deshabilita cerrar

        # Frame for text and scrollbar
        frame = ttk.Frame(loading_win)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        text_widget = tk.Text(frame, height=3, wrap="word")
        text_widget.insert("1.0", message)
        text_widget.config(state="disabled")
        text_widget.pack(side="left", fill="both", expand=True)

        # Horizontal scrollbar
        x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text_widget.xview)
        text_widget.configure(xscrollcommand=x_scroll.set)
        x_scroll.pack(side="bottom", fill="x")
        return loading_win
    except Exception as e:
        logging.exception("Error showing loading modal: %s", e)
        raise e


def show_loading_modal(
    parent: tk.Tk,
    on_call: Callable,
    args: tuple,
    message: str = "Loading...",
    keep_open: bool = False,
    log_queue_poll: Callable[[], int] | None = None,
) -> None:
    """Show a modal loading window with logging output redirected to it.

    Args:
        parent: Parent widget
        on_call: Function to call
        args: Arguments to pass to on_call
        message: Loading message to display
        keep_open: Whether to keep the progress window open after completion
        log_queue_poll: Optional function to poll child process log queue
    """
    loading_win = _create_loading_modal(parent, message)
    on_call_wrapped = _call_wrapper(parent, on_call, loading_win, keep_open)
    t = threading.Thread(target=on_call_wrapped, args=args)
    t.start()
    while t.is_alive():
        # Poll log queue from child processes if provided
        if log_queue_poll is not None:
            try:
                log_queue_poll()
            except Exception:
                pass
        parent.update()
        # time.sleep(0.01)  # Optional small sleep to reduce CPU usage


def auto_close_info(parent, title, message, timeout=5000) -> None:
    """Show an info dialog that auto-closes after a timeout."""
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("300x120")
    win.transient(parent)
    win.grab_set()
    label = ttk.Label(win, text=message, anchor="center", padding=20)
    label.pack(expand=True, fill="both")
    btn = ttk.Button(win, text="OK", command=win.destroy)
    btn.pack(pady=(0, 10))
    win.after(timeout, win.destroy)
    win.protocol("WM_DELETE_WINDOW", win.destroy)
    while win.winfo_exists():
        parent.after(100, parent.update())
