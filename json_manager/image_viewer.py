"""Image viewer frame for displaying images in the GUI."""

import os
import logging
from typing import Callable, cast
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from json_gui.json_manager.scroll_utils import bind_frame_scroll_events
from json_gui.json_manager.json_tree_editor import open_preview


class ImageViewer(ttk.Frame):
    """Frame for displaying images."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.images: list[ImageTk.PhotoImage] = []  # Keep references

        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#2b2b2b")
        self.scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel only when mouse is over this widget
        bind_frame_scroll_events(self, self.canvas, True)

    def display_images(self, image_paths: list[str]) -> None:
        """Display images from file paths."""
        self.images.clear()
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for _i, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                # Resize to fit
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.images.append(photo)

                frame = ttk.Frame(self.scrollable_frame)
                frame.pack(side="left", padx=10, pady=10)

                label = ttk.Label(frame, image=photo)
                label.pack()

                name_label = ttk.Label(frame, text=os.path.basename(path), wraplength=400)
                name_label.pack()

                # if click frame, open image in default viewer
                for widget in (frame, label, name_label):
                    callback: Callable[[tk.Event], None] = cast(
                        Callable[[tk.Event], None], lambda e, p=path, f=frame: open_preview(p, f)
                    )
                    widget.bind("<Button-1>", callback)

            except Exception as e:
                error_label = ttk.Label(self.scrollable_frame, text=f"Error loading {path}: {e}")
                error_label.pack(side="left", padx=10, pady=10)
                logging.exception("Error loading image %s", path)

    def clear(self) -> None:
        """Clear all images."""
        self.images.clear()
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
