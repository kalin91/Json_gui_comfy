"""
Mocked PromptServer for testing purposes in Impact Pack.
It'll display a simple Tkinter messagebox while mocked server is starting.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable
import utils as _  # noqa: F401


def show_loading(parent, message="Loading MockServer...") -> tk.Toplevel:
    """Show a loading window with a message."""
    win = tk.Toplevel(parent)
    win.title("Loading")
    win.geometry("300x80")
    win.transient(parent)
    win.grab_set()
    label = tk.Label(win, text=message, font=("TkDefaultFont", 12))
    label.pack(expand=True, fill="both", padx=20, pady=20)
    win.update()
    return win


# Mock PromptServer for Impact Pack
class MockServer:
    """
    A mocked PromptServer for testing purposes in Impact Pack.
    """

    def __init__(self):
        self.routes = self
        self.last_node_id = "mock_node_id"

    def post(self, _route) -> Callable:
        """Mocked post decorator."""

        def decorator(func) -> Callable:
            """Return the original function without modification."""
            return func

        return decorator

    def get(self, _route) -> Callable:
        """Mocked get decorator."""

        def decorator(func) -> Callable:
            """Return the original function without modification."""
            return func

        return decorator

    def add_on_prompt_handler(self, handler) -> None:
        """Mocked add_on_prompt_handler method."""

    def send_sync(self, event, data, sid=None) -> None:
        """Mocked send_sync method."""


root = tk.Tk()
style = ttk.Style()
if "clam" in style.theme_names():
    style.theme_use("clam")

# Show loading window
loading_win = show_loading(root)

import server  # noqa: E402 pylint: disable=C0413

# Slow initialization goes here
server.PromptServer.instance = MockServer()

# Close loading window
loading_win.destroy()
root.destroy()
