""" "JSON Tree Editor GUI Component."""

import math
import re
import os
import sys
import subprocess
import logging
import tkinter as tk
import random
from tkinter import ttk, messagebox
from typing import Any, Callable, Optional, cast
from PIL import Image, ImageTk
import json_gui.utils as gui_utils
from json_gui.typedicts import BodyDict, is_bodydict
from json_gui.json_manager.scroll_utils import bind_frame_scroll_events, bind_scroll_events
from json_gui.constants import get_combo_constants, JSON_CANVAS_NAME, JSON_SCROLL_FRAME_NAME


class StickyCanvas(tk.Canvas):
    """A Canvas that supports sticky headers/columns."""

    _image: Optional[ImageTk.PhotoImage]

    @property
    def image(self) -> Optional[ImageTk.PhotoImage]:
        """Get the current image."""
        return self._image

    @image.setter
    def image(self, value: ImageTk.PhotoImage) -> None:
        """Set the current image."""
        self._image = value

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._image = None


def open_preview(file_path: str, frame: ttk.Widget) -> None:
    """Open a floating preview window for the selected image."""
    try:
        try:
            img = Image.open(file_path)
        except Exception as e:
            messagebox.showerror("Preview Error", f"Cannot open image:\n{e}")
            return

        parent_win = frame.winfo_toplevel()
        win = tk.Toplevel(parent_win)
        win.title(f"Preview - {os.path.basename(file_path)}")
        win.transient(parent_win)
        win.resizable(True, True)

        # Container for canvas and scrollbars
        container = tk.Frame(win, background="blue")
        win.update_idletasks()  # Asegura que la ventana esté dibujada
        container.pack(fill="both", expand=True)

        v_scroll = ttk.Scrollbar(container, orient="vertical")
        h_scroll = ttk.Scrollbar(container, orient="horizontal")

        # Compute max preview size relative to screen
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        max_w = int(sw * 0.9)
        max_h = int(sh * 0.9)

        img_w, img_h = img.size

        # Calculate window size (image size + padding, clamped to max screen size)
        scale_ratio: float = min(max_w / img_w, max_h / img_h, 1)
        win_w = math.ceil((img_w * scale_ratio) + v_scroll.winfo_reqwidth())
        win_h = math.ceil((img_h * scale_ratio) + h_scroll.winfo_reqheight())

        win.geometry(f"{win_w}x{win_h}")

        # Variables for zoom
        original_img = img.copy()

        if scale_ratio < 1.0:
            new_w = math.floor(img_w * scale_ratio)
            new_h = math.floor(img_h * scale_ratio)
            try:
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            except Exception:
                img = img.resize((new_w, new_h))

        canvas = StickyCanvas(container, highlightthickness=0, yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        v_scroll.config(command=canvas.yview)
        h_scroll.config(command=canvas.xview)

        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        photo = ImageTk.PhotoImage(img)
        image_item = canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo

        # Set initial scrollregion
        canvas.config(scrollregion=(0, 0, img.width, img.height))

        def zoom(event) -> str:
            """Zoom in or out on the image."""
            nonlocal scale_ratio
            if event.delta > 0 or event.num == 4:  # Zoom in
                scale_ratio *= 1.1
            elif event.delta < 0 or event.num == 5:  # Zoom out
                scale_ratio /= 1.1

            # Limit zoom
            scale_ratio = max(0.1, min(scale_ratio, 5.0))

            new_w = int(original_img.width * scale_ratio)
            new_h = int(original_img.height * scale_ratio)

            try:
                resized = original_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            except Exception:
                resized = original_img.resize((new_w, new_h))

            new_photo = ImageTk.PhotoImage(resized)
            canvas.itemconfig(image_item, image=new_photo)
            canvas.image = new_photo
            canvas.config(scrollregion=(0, 0, new_w, new_h))
            return "break"

        # Bind zoom events
        # Bind to canvas specifically to override scroll bindings
        canvas.bind("<Control-MouseWheel>", zoom)
        canvas.bind("<Control-Button-4>", zoom)
        canvas.bind("<Control-Button-5>", zoom)

        # Also bind to window just in case focus is elsewhere
        win.bind("<Control-MouseWheel>", zoom)  # Windows
        win.bind("<Control-Button-4>", zoom)  # Linux scroll up
        win.bind("<Control-Button-5>", zoom)  # Linux scroll down

        # Bind scroll events for panning (Shift+Scroll for horizontal)
        bind_frame_scroll_events(canvas, canvas)

        def copy_to_clipboard(_event=None) -> str:
            """Copy image to clipboard."""
            try:
                if sys.platform.startswith("linux"):
                    try:
                        subprocess.run(
                            ["xclip", "-selection", "clipboard", "-t", "image/png", "-i", file_path], check=True
                        )
                        messagebox.showinfo("Info", "Image copied to clipboard")
                        return "break"
                    except FileNotFoundError:
                        pass

                    try:
                        with open(file_path, "rb") as f:
                            subprocess.run(["wl-copy"], input=f.read(), check=True)
                        messagebox.showinfo("Info", "Image copied to clipboard")
                        return "break"
                    except FileNotFoundError:
                        pass

                    messagebox.showwarning("Warning", "Install xclip or wl-copy to copy images on Linux.")

                elif sys.platform == "win32":
                    safe_path = file_path.replace("'", "''")
                    cmd = f"Set-Clipboard -Path '{safe_path}'"
                    subprocess.run(["powershell", "-command", cmd], check=True)
                    messagebox.showinfo("Info", "Image copied to clipboard")
                    return "break"

            except Exception as e:
                logging.exception("Failed to copy to clipboard")
                messagebox.showerror("Error", f"Failed to copy to clipboard:\n{e}")
            return "break"

        win.bind("<Control-c>", copy_to_clipboard)
    except Exception as e:
        logging.exception("Error opening preview window: %s", e)
        messagebox.showerror("Preview Error", f"Error opening preview:\n{e}")


def _create_string_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    notify_change: Callable[[], None],
    string_entries: dict[str, tk.Entry],
) -> None:
    """Create a string entry widget."""
    try:
        entry = ttk.Entry(frame, width=60)
        entry.insert(0, str(value))
        entry.bind("<KeyRelease>", lambda e: notify_change())
        entry.pack(side="left", padx=5, fill="x", expand=True)
        assert isinstance(value, str), f"Value for key '{key}' must be a string"
        string_entries[full_key] = entry
    except Exception as e:
        logging.exception("Error creating string entry for key '%s': %s", key, e)


def _create_boolean_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    notify_change: Callable[[], None],
    boolean_vars: dict[str, tk.BooleanVar],
) -> None:
    """Create a boolean entry widget."""
    assert isinstance(value, bool), f"Value for key '{key}' must be a bool in list item '{key}'"
    try:
        var = tk.BooleanVar(value=value)
        check = ttk.Checkbutton(frame, variable=var, command=notify_change)
        check.pack(side="left", padx=5)
        boolean_vars[full_key] = var
    except Exception as e:
        logging.exception("Error creating boolean entry for key '%s': %s", key, e)
        raise e


def _create_open_preview_handler(p_combo: ttk.Combobox, p_folder: str, p_frame: ttk.Widget) -> Callable[[], None]:
    """Create a handler to open a preview window."""

    def _open_preview(folder=p_folder, combo=p_combo, frame=p_frame) -> None:
        """Open a floating preview window for the selected image."""
        combo_value = combo.get()
        path = os.path.join(folder, combo_value) if combo_value and combo_value != "<None>" else ""
        if not path:
            messagebox.showwarning("Preview", "Select a file to preview")
            return
        open_preview(path, frame)

    return _open_preview


def _create_file_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    body: dict[str, Any],
    notify_change: Callable[[], None],
    file_entries: dict[str, ttk.Combobox],
) -> None:
    """Create a file entry widget."""
    try:
        assert isinstance(value, str), f"Value for key '{key}' must be a string"
        assert "parent" in body[key], f"'parent' not specified for file type key '{key}' in body"
        body_parent = body[key]["parent"]
        assert isinstance(body_parent, str), f"'parent' for file type key '{key}' must be a string"
        combo = ttk.Combobox(frame, width=57, state="readonly")
        files: list[str]
        folder: str
        combo.bind("<<ComboboxSelected>>", lambda e: notify_change())

        # Bind mousewheel directly to text widget (not bind_all)
        bind_frame_scroll_events(combo, combo)

        combo.pack(side="left", padx=5, fill="x", expand=True)
        if body_parent == "input":
            files, folder = gui_utils.get_input_files_recursive()
            ttk.Button(frame, text="Preview", command=_create_open_preview_handler(combo, folder, frame)).pack(
                side="left", padx=(5, 5)
            )
        else:
            body_prefix = body[key].get("prefix", "")
            files, folder = gui_utils.get_folder_files_recursive(body_parent)
            files = [f"{body_prefix}{f}" for f in files]
        if body[key].get("hasNone", False):
            files.insert(0, "<None>")
        combo["values"] = files
        if value in files:
            combo.set(value)
        file_entries[full_key] = combo
    except Exception as e:
        logging.exception("Error creating file entry for key '%s': %s", key, e)
        raise e


def _create_combo_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    body: dict[str, Any],
    notify_change: Callable[[], None],
    combo_entries: dict[str, ttk.Combobox],
) -> None:
    """Create a combo entry widget."""
    try:
        assert isinstance(value, str), f"Value for key '{key}' must be a string"
        assert (
            "constant" in body[key] or "values" in body[key]
        ), f"'constant' or 'values' not specified for combo type key '{key}' in body"
        combo_values: list[str]
        if "values" in body[key]:
            combo_values = body[key]["values"]
        else:
            assert body[key]["constant"] in get_combo_constants(), (
                f"Constant '{body[key]['constant']}' not found " f"in constants dictionary for combo type key '{key}'"
            )
            combo_values = get_combo_constants()[body[key]["constant"]]
        combo = ttk.Combobox(frame, width=57, state="readonly")
        combo["values"] = combo_values
        if value in combo_values:
            combo.set(value)
        combo.bind("<<ComboboxSelected>>", lambda e: notify_change())

        # Bind mousewheel directly to text widget (not bind_all)
        bind_frame_scroll_events(combo, combo)

        combo.pack(side="left", padx=5, fill="x", expand=True)
        combo_entries[full_key] = combo
    except Exception as e:
        logging.exception("Error creating combo entry for key '%s': %s", key, e)
        raise e


def _create_multiline_text_widget(
    parent: tk.Widget,
    key: str,
    value: Any,
    full_key: str,
    indent: int,
    notify_change: Callable[[], None],
    text_entries: dict[str, tk.Text],
) -> None:
    """Create a multiline text widget."""
    try:
        assert isinstance(value, str), f"Value for key '{key}' must be a string"
        # Multiline text box for positive/negative prompts

        text_frame = ttk.Frame(parent)
        text_frame.pack(fill="x", padx=(indent * 20 + 10, 5), pady=5)

        text_widget = tk.Text(text_frame, height=8, width=80, wrap="word")
        text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=text_scrollbar.set)

        text_widget.insert("1.0", str(value))
        text_widget.bind("<KeyRelease>", lambda e: notify_change())

        # Bind mousewheel directly to text widget (not bind_all)
        bind_scroll_events(text_widget)

        text_widget.pack(side="left", fill="both", expand=True)
        text_scrollbar.pack(side="right", fill="y")

        text_entries[full_key] = text_widget
    except Exception as e:
        logging.exception("Error creating multiline text widget for key '%s': %s", key, e)
        raise e


def _create_number_validator(
    min_val: float, max_val: float, type_val: type, last_val: tuple[list[Any], list[ttk.Spinbox]], format_str: str
) -> Callable[[str], bool]:
    """ "Create a number validator function."""

    def validate_number(
        value: str,
        p_min=min_val,
        p_max=max_val,
        t_val: type = type_val,
        l_val: tuple[list[Any], list[ttk.Spinbox]] = last_val,
        p_format: str = format_str,
    ) -> bool:
        """Validate number input."""
        try:
            last_value = l_val[0][0] if l_val[0] else None
            assert last_value or last_value == 0, "Last value not found for validation"
            entry_widget: ttk.Spinbox = l_val[1][0]
            assert entry_widget, "Entry widget not found for validation"

            def reset_value(to_val: Optional[str] = None) -> None:
                """Reset entry to last valid value."""
                to_val = str(last_value) if to_val is None else to_val
                entry_widget.set(to_val)
                entry_widget.config(validate="focusout")

            val_zero: bool = False
            if value in ("", "-", "."):
                value = "0"
                val_zero = True

            val = t_val(value)
            if val < p_min or val > p_max:
                entry_widget.after_idle(reset_value)
                return False
            if t_val == float and "." in value:
                formatted_val = p_format % val
                max_decimals: int = len(formatted_val.split(".")[-1])
                actual_decimals: int = len(value.split(".")[-1])
                if actual_decimals > max_decimals:
                    entry_widget.after_idle(reset_value)
                    return False
            l_val[0][0] = val
            if val_zero:
                entry_widget.after_idle(lambda: reset_value(val))
            else:
                entry_widget.after_idle(lambda: entry_widget.config(validate="focusout"))
            return True
        except ValueError:
            entry_widget.after_idle(reset_value)
            return False
        except Exception:
            logging.exception("Unexpected error during validation")
            entry_widget.after_idle(reset_value)
            return False

    return validate_number


def _create_on_invalid_handler(
    body_type: str, min_val: float, max_val: float, format_str: str, notify_change: Callable[[], None]
) -> Callable[[], None]:
    """Create an invalid input handler."""

    def on_invalid(
        b_type=body_type, p_min=min_val, p_max=max_val, p_format=format_str, on_change=notify_change
    ) -> None:
        """Handle invalid input."""
        try:
            max_decimals = 0 if b_type == "int" else int(p_format.replace("f", "").split(".")[-1])
            messagebox.showwarning(
                "Invalid Input",
                (
                    f"Please enter a valid {b_type} between {p_min} and "
                    f"{p_max} with up to {max_decimals} decimal places."
                ),
            )
            on_change()
        except Exception as e:
            logging.exception("Error handling invalid input: %s", e)
            on_change()
            raise e

    return on_invalid


def _create_randomize_handler(
    entry: ttk.Spinbox,
    min_val: float,
    max_val: float,
    format_str: str,
    body_type: str,
    notify_change: Callable[[], None],
) -> Callable[[], None]:
    """Create a randomize button handler."""

    def set_random(e=entry, mn=min_val, mx=max_val, fmt=format_str, bt=body_type, on_change=notify_change) -> None:
        """Set a random value in the entry."""
        try:
            on_change()
            if bt == "int":
                vali = random.randint(int(mn), int(mx))
                e.set(vali)
            else:
                valf = random.uniform(float(mn), float(mx))
                e.set(fmt % valf)
        except Exception as ex:
            logging.exception("Error setting random value")
            raise ex

    return set_random


def _create_numeric_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    register_call: Callable[[Callable[..., Any]], str],
    notify_change: Callable[[], None],
    body: dict[str, Any],
    body_type: str,
    full_key: str,
    int_entries: dict[str, tk.Entry],
    float_entries: dict[str, tk.Entry],
) -> None:
    """Create a numeric entry widget."""
    try:
        type_val: type = int if body_type == "int" else float
        min_val: float = body[key].get("min", -999999999999999)
        max_val: float = body[key].get("max", 999999999999999)
        format_str: str = "%.0f" if body_type == "int" else body[key].get("format", "%.1f")
        last_val: tuple[list[Any], list[ttk.Spinbox]] = ([value], [])

        entry = ttk.Spinbox(
            frame,
            from_=min_val,
            to=max_val,
            increment=body[key].get("step", 1.0),
            width=25,
            wrap=True,
            format=format_str,
            command=notify_change,
            validate="focusout",
            validatecommand=(
                register_call(_create_number_validator(min_val, max_val, type_val, last_val, format_str)),
                "%P",
            ),
            invalidcommand=(
                register_call(_create_on_invalid_handler(body_type, min_val, max_val, format_str, notify_change)),
            ),
        )
        entry.set(type_val(value))
        last_val[1].append(entry)
        entry.bind("<KeyRelease>", lambda e: notify_change())

        # Bind mousewheel directly to text widget (not bind_all)
        bind_frame_scroll_events(entry, entry)

        entry.pack(side="left", padx=(0, 5))
        if body[key].get("randomizable", False):
            entry.config(foreground="blue")
            ttk.Button(
                frame,
                text="Random",
                command=_create_randomize_handler(entry, min_val, max_val, format_str, body_type, notify_change),
            ).pack(side="left", padx=(0, 5))

            assert isinstance(type_val(value), type_val), f"Value for key '{key}' must be an {body_type}"
        if body_type == "int":
            int_entries[full_key] = entry
        elif body_type == "float":
            float_entries[full_key] = entry
    except Exception as e:
        logging.exception("Error creating numeric entry for key '%s': %s", key, e)
        raise e


class JSONTreeEditor(ttk.Frame):
    """A hierarchical, editable view for JSON data."""

    _body: Optional[BodyDict]

    @property
    def body(self) -> BodyDict:
        """Get the body definition."""
        if self._body is None:
            raise ValueError("Try to access body when it is not set")
        return self._body

    @body.setter
    def body(self, value: BodyDict) -> None:
        """Set the body definition."""
        if not is_bodydict(value):
            raise ValueError("body must conform to BodyDict structure")
        self._body = value

    def __init__(self, parent: tk.Widget, on_change: Callable[[bool], None], on_refresh: Callable[[], bool]) -> None:
        super().__init__(parent)
        self.data: dict[str, Any] = {}
        self._body = None
        self.string_entries: dict[str, tk.Entry] = {}
        self.int_entries: dict[str, tk.Entry] = {}
        self.float_entries: dict[str, tk.Entry] = {}
        self.text_entries: dict[str, tk.Text] = {}  # For multiline text widgets
        self.boolean_vars: dict[str, tk.BooleanVar] = {}
        self.list_entries: dict[str, list[dict[str, Any]]] = {}
        self.file_entries: dict[str, ttk.Combobox] = {}
        self.combo_entries: dict[str, ttk.Combobox] = {}
        self._on_change: Callable[[bool], None] = on_change  # Callback when any value changes
        self._on_refresh = on_refresh  # Callback to check for unsaved changes

        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, name=JSON_CANVAS_NAME)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, name=JSON_SCROLL_FRAME_NAME)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Add horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        # Use grid layout for proper scrollbar sizing
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")

        # Bind mousewheel only when mouse is over this widget
        bind_frame_scroll_events(self, self.canvas, True)

    def load_data(self, data: dict[str, Any], body: BodyDict) -> None:
        """Load JSON data into the editor."""
        self.data = data
        self.body = body
        self.string_entries.clear()
        self.int_entries.clear()
        self.float_entries.clear()
        self.text_entries.clear()
        self.boolean_vars.clear()
        self.list_entries.clear()
        self.file_entries.clear()
        self.combo_entries.clear()

        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        assert "props" in body, "'props' key not found in body"

        self._build_tree(self.scrollable_frame, data, body["props"], "")
        # Use after_idle to ensure all widget events are processed before marking as clean
        self.after_idle(lambda: self._on_change(False))

    def _create_default_item(self, body_def: dict[str, Any]) -> Any:
        """Create a default item based on body definition. Uses empty/minimal values."""
        body_type = body_def.get("type", "string")
        if body_type == "object":
            props = body_def.get("props", {})
            result: dict[str, Any] = {}
            for prop_key, prop_def in props.items():
                if prop_def.get("isArray", False):
                    result[prop_key] = []
                else:
                    result[prop_key] = self._create_default_item(prop_def)
            return result
        elif body_type in ("string", "file", "combo", "multiline_string"):
            return ""
        elif body_type == "int":
            return body_def.get("min", 0)
        elif body_type == "float":
            return body_def.get("min", 0.0)
        elif body_type == "bool":
            return False
        else:
            return ""

    def _add_array_item(self, key: str, body_def: dict[str, Any], is_first: bool) -> None:
        """Add an item to an array at the specified position."""

        if not self._on_refresh():
            return

        # Create new item based on body definition
        new_item = self._create_default_item(body_def)

        # Get current list or create if not exists
        data_list: list = self._set_nested_value(self.data, key, [])

        # Add at position
        if is_first:
            data_list.insert(0, new_item)
        else:
            data_list.append(new_item)

        # Update the data structure
        self._set_nested_value(self.data, key, data_list)

        # Reload the tree
        self.load_data(self.data, self.body)
        self.after_idle(self._notify_change)

    def _delete_array_item(self, key: str, index: int) -> None:
        """Delete an item from an array at the specified index."""

        if not self._on_refresh():
            return

        # Get current list or create if not exists
        data_list: list = self._set_nested_value(self.data, key, [])

        # Delete item
        if 0 <= index < len(data_list):
            del data_list[index]

        # Update the data structure
        self._set_nested_value(self.data, key, data_list)

        # Reload the tree
        self.load_data(self.data, self.body)
        self.after_idle(self._notify_change)

    def _build_tree(
        self, parent: tk.Widget, data: dict[str, Any], body: dict[str, Any], prefix: str, indent: int = 0
    ) -> None:
        """Recursively build the tree view."""
        # List of keys that should use multiline text boxes
        try:
            for key, value in data.items():
                assert key in body, f"'{key}' key not found in body"
                assert isinstance(body[key], dict), f"Body for key '{key}' must be a dict"
                assert "type" in body[key], f"'type' not specified for key '{key}' in body"
                body_type: str = body[key]["type"]
                assert "isArray" in body[key], f"'isArray' not specified for key '{key}' in body"
                body_is_array: bool = body[key]["isArray"]
                full_key = f"{prefix}.{key}" if prefix else key
                frame = ttk.Frame(parent)
                frame.pack(fill="x", padx=(indent * 20, 5), pady=2)

                if body_is_array:
                    assert isinstance(value, list), f"Value for key '{key}' must be a list"
                    item_body_def = body[key]  # Body definition for array items

                    # Array header with label and add buttons
                    label = ttk.Label(frame, text=f"▼ {key} (Array):", font=("TkDefaultFont", 10, "bold"))
                    label.pack(side="left")

                    # Add first button - pass the list reference and body definition
                    add_first_btn = tk.Button(
                        frame,
                        text="+ First",
                        font=("Arial", 7),
                        command=cast(Callable[[], None],lambda k=full_key, bdef=item_body_def: self._add_array_item(k, bdef, True),)
                    )
                    add_first_btn.pack(side="left", padx=(10, 2))

                    # Add last button
                    add_last_btn = tk.Button(
                        frame,
                        text="+ Last",
                        font=("Arial", 7),
                        command=cast(Callable[[], None],lambda k=full_key, bdef=item_body_def: self._add_array_item(k, bdef, False),)
                    )
                    add_last_btn.pack(side="left", padx=2)

                    for i, item in enumerate(value):
                        item_key = f"{full_key}[{i}]"
                        item_frame = ttk.Frame(parent)
                        item_frame.pack(fill="x", padx=((indent + 1) * 20, 5), pady=2)

                        # Item label
                        item_label = ttk.Label(item_frame, text=f"{key} [{i}]:", font=("TkDefaultFont", 10, "italic"))
                        item_label.pack(side="left")

                        # Delete button for this item - pass the list reference
                        delete_btn = tk.Button(
                            item_frame,
                            text="Delete",
                            font=("Arial", 7),
                            fg="red",
                            command=cast(Callable[[], None],lambda k=full_key, idx=i: self._delete_array_item(k, idx)),
                        )
                        delete_btn.pack(side="left", padx=(10, 0))

                        if body_type == "object":
                            assert isinstance(item, dict), f"List item '{key}' must be a dict"
                            assert "props" in body[key], f"'props' not specified for key '{key}' in body"
                            self._build_tree(parent, item, body[key]["props"], item_key, indent + 3)
                        else:
                            # Primitive types in list - create a temporary body without isArray
                            temp_body = {f"{key}_{i}": {k: v for k, v in body[key].items() if k != "isArray"}}
                            temp_body[f"{key}_{i}"]["isArray"] = False
                            self._build_tree(parent, {f"{key}_{i}": item}, temp_body, item_key, indent + 3)
                elif body_type == "object":
                    assert isinstance(value, dict), f"Value for key '{key}' must be a dict"
                    assert "props" in body[key], f"'props' not specified for key '{key}' in body"
                    # Expandable section for dict
                    label = ttk.Label(frame, text=f"▼ {key}:", font=("TkDefaultFont", 10, "bold"))
                    label.pack(anchor="w")
                    self._build_tree(parent, value, body[key]["props"], full_key, indent + 1)
                else:
                    label_font: tuple = ("TkDefaultFont", 10)
                    if indent == 0:
                        label_font = label_font + ("bold",)
                    label = ttk.Label(frame, text=f"{key}:", font=label_font)
                    label.pack(side="left")

                    if body_type == "bool":
                        _create_boolean_entry(
                            frame,
                            key,
                            value,
                            full_key,
                            self._notify_change,
                            self.boolean_vars,
                        )
                    elif body_type == "multiline_string":
                        _create_multiline_text_widget(
                            parent,
                            key,
                            value,
                            full_key,
                            indent,
                            self._notify_change,
                            self.text_entries,
                        )

                    elif body_type in ("string", "float", "int"):
                        if body_type == "string":
                            _create_string_entry(
                                frame,
                                key,
                                value,
                                full_key,
                                self._notify_change,
                                self.string_entries,
                            )
                        elif body_type in ("int", "float"):
                            _create_numeric_entry(
                                frame,
                                key,
                                value,
                                self.register,
                                self._notify_change,
                                body,
                                body_type,
                                full_key,
                                self.int_entries,
                                self.float_entries,
                            )
                        else:
                            raise ValueError(f"Unsupported body type: {body_type}")
                    elif body_type == "file":
                        _create_file_entry(
                            frame,
                            key,
                            value,
                            full_key,
                            body,
                            self._notify_change,
                            self.file_entries,
                        )
                    elif body_type == "combo":
                        _create_combo_entry(
                            frame,
                            key,
                            value,
                            full_key,
                            body,
                            self._notify_change,
                            self.combo_entries,
                        )
                    else:
                        raise ValueError(f"Unsupported body type: {body_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Error building JSON tree at prefix '{prefix}':\n{e}")
            logging.exception("Error building JSON tree at prefix '%s': %s", prefix, e)
            raise e

    def get_data(self) -> dict[str, Any]:
        """Get the current data from the editor."""
        result = self._deep_copy_structure(self.data)

        # Update string entries
        for full_key, entry in self.string_entries.items():
            self._set_nested_value(result, full_key, self._parse_value(entry.get()))

        # Update int entries
        for full_key, entry in self.int_entries.items():
            self._set_nested_value(result, full_key, self._parse_value(entry.get()))

        # Update float entries
        for full_key, entry in self.float_entries.items():
            self._set_nested_value(result, full_key, self._parse_value(entry.get()))

        # Update multiline text entries
        for full_key, text_widget in self.text_entries.items():
            text_value = text_widget.get("1.0", "end-1c")  # Get text without trailing newline
            self._set_nested_value(result, full_key, text_value)

        # Update boolean entries
        for full_key, var in self.boolean_vars.items():
            self._set_nested_value(result, full_key, var.get())

        # Update file entries
        for full_key, combo in self.file_entries.items():
            self._set_nested_value(result, full_key, combo.get())

        # Update combo entries
        for full_key, combo in self.combo_entries.items():
            self._set_nested_value(result, full_key, combo.get())

        # Update internal data copy
        self.data = self._deep_copy_structure(result)

        return result

    def _deep_copy_structure(self, obj: Any) -> Any:
        """Deep copy maintaining structure."""
        if isinstance(obj, dict):
            return {k: self._deep_copy_structure(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy_structure(item) for item in obj]
        else:
            return obj

    def _set_nested_value(self, data: dict, key_path: str, value: Any) -> Any:
        """Set a value in a nested dict using dot notation and returns previous value."""
        try:
            old_value: Any = None
            keys = key_path.split(".")
            current = data
            for key in keys[:-1]:
                # Handle list indices
                match = re.match(r"(\w+)\[(\d+)\]", key)
                if match:
                    list_key = match.group(1)
                    index = int(match.group(2))
                    assert list_key in current and isinstance(
                        current[list_key], list
                    ), f"List key '{list_key}' not found in data"
                    next_current = current[list_key][index]
                    if not isinstance(next_current, dict):
                        old_value = next_current
                        current[list_key][index] = value
                        continue
                    current = next_current
                if key in current:
                    current = current[key]
            final_key = keys[-1]
            if final_key in current:
                old_value = current[final_key]
                current[final_key] = value
            return old_value
        except Exception as e:
            messagebox.showerror("Error", f"Error setting nested value for key '{key_path}':\n{e}")
            logging.exception("Error setting nested value for key '%s': %s", key_path, e)
            raise e

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value to its appropriate type.

        Note: Boolean conversion is NOT done here because booleans are handled
        separately via BooleanVar checkboxes. Strings like "True" or "False"
        should remain as strings.
        """
        if value_str.lower() == "null" or value_str.lower() == "none":
            return None
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            return value_str

    def _notify_change(self) -> None:
        """Notify that a value has changed."""
        self._on_change(True)
