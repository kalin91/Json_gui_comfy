"""Utility functions for handling scroll events in Tkinter widgets."""

import logging
from typing import Callable
import tkinter as tk
from json_gui.constants import JSON_CANVAS_NAME


def _on_mousewheel(widget: tk.YView, event: tk.Event) -> str:
    """Handle mousewheel in text widget without propagating to parent."""
    try:
        if event.num == 4:
            widget.yview_scroll(-1, "units")
        elif event.num == 5:
            widget.yview_scroll(1, "units")
        else:
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"  # Prevent event propagation
    except Exception as e:
        logging.exception("Error handling mousewheel event: %s", e)
        raise e


def _on_shift_mousewheel(widget: tk.XView, event: tk.Event) -> str:
    """Handle horizontal mousewheel in text widget."""
    try:
        if event.num == 4:
            widget.xview_scroll(-1, "units")
        elif event.num == 5:
            widget.xview_scroll(1, "units")
        else:
            widget.xview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"  # Prevent event propagation
    except Exception as e:
        logging.exception("Error handling shift mousewheel event: %s", e)
        raise e


def bind_scroll_events(widget: tk.Widget, bind_all: bool = False) -> None:
    """Bind scroll events to a widget so scrolling works when hovering over it."""

    try:
        bind_call: Callable[[str, Callable[[tk.Event], str]], str] = widget.bind_all if bind_all else widget.bind

        if not bind_all:
            _unbind_scroll_events(widget, True)

        if isinstance(widget, tk.YView):
            on_wheel: Callable[[tk.Event], str] = lambda event: _on_mousewheel(widget, event)
            # Windows/MacOS
            bind_call("<MouseWheel>", on_wheel)
            # Linux scroll up/down
            bind_call("<Button-4>", on_wheel)
            bind_call("<Button-5>", on_wheel)
        if isinstance(widget, tk.XView):
            on_shift_wheel: Callable[[tk.Event], str] = lambda event: _on_shift_mousewheel(widget, event)
            # Shift+scroll for horizontal (Windows/MacOS)
            bind_call("<Shift-MouseWheel>", on_shift_wheel)
            # Linux horizontal scroll
            bind_call("<Shift-Button-4>", on_shift_wheel)
            bind_call("<Shift-Button-5>", on_shift_wheel)
    except Exception as e:
        logging.exception("Error binding scroll events: %s", e)
        raise e


def _unbind_scroll_events(widget: tk.Widget, unbind_all: bool = False) -> None:
    """Unbind mousewheel events when mouse leaves the scroll area."""
    try:
        unbind_call: Callable[[str], None] = widget.unbind_all if unbind_all else widget.unbind

        # Windows/MacOS
        unbind_call("<MouseWheel>")
        # Linux scroll up/down
        unbind_call("<Button-4>")
        unbind_call("<Button-5>")
        # Shift+scroll for horizontal (Windows/MacOS)
        unbind_call("<Shift-MouseWheel>")
        # Linux horizontal scroll
        unbind_call("<Shift-Button-4>")
        unbind_call("<Shift-Button-5>")

        if not unbind_all:
            canvas_str_pos: int = widget.winfo_parent().rfind(JSON_CANVAS_NAME)
            if canvas_str_pos != -1:
                canvas_name: str = widget.winfo_parent()[0:canvas_str_pos] + JSON_CANVAS_NAME
                canvas: tk.Widget = widget.nametowidget(canvas_name)
                bind_scroll_events(canvas, True)
    except Exception as e:
        logging.exception("Error unbinding scroll events: %s", e)
        raise e


def bind_frame_scroll_events(hover_widget: tk.Widget, event_target: tk.Widget, bind_all: bool = False) -> None:
    """Bind scroll events to a widget so scrolling works when hovering over it."""
    try:
        enter_call: Callable[[tk.Event], None] = lambda event: bind_scroll_events(event_target, bind_all)
        leave_call: Callable[[tk.Event], None] = lambda event: _unbind_scroll_events(event_target, bind_all)
        hover_widget.bind("<Enter>", enter_call)
        hover_widget.bind("<Leave>", leave_call)
    except Exception as e:
        logging.exception("Error binding scroll frame events: %s", e)
        raise e
