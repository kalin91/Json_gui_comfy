"""GUI Application for managing JSON configuration files."""

from importlib.util import spec_from_file_location, module_from_spec
import json
import os
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Type, cast
import uuid
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import yaml
import folder_paths
from json_gui.json_manager.json_tree_editor import JSONTreeEditor
from json_gui.json_manager.scroll_utils import bind_frame_scroll_events
from json_gui.json_manager.image_viewer import ImageViewer
from json_gui.json_manager import loading_modal
from json_gui import p_logger, utils as gui_utils
from json_gui.typedicts import BodyDict, is_bodydict
from json_gui.json_manager.memory_utils import (
    check_memory_available,
    manual_cleanup,
    cleanup_vram,
    check_memory_fragmentation,
    show_memory_details,
)


class JSONManagerApp:
    """Main application class."""

    _flow_body: Optional[BodyDict]
    _flow_inst: Optional[gui_utils.AbsFlow] = None

    # Update scrollregion and frame width when content or canvas changes
    def _update_actions_scrollregion(
        self,
        parent: ttk.Widget,
        p_canvas: tk.Canvas,
        p_frame: ttk.Widget,
        p_window_id: int,
        _event: tk.Event | None = None,
    ) -> None:
        p_frame.update_idletasks()
        content_width = p_frame.winfo_reqwidth()
        canvas_width = p_canvas.winfo_width()
        # Use the larger of content width or canvas width
        final_width = max(content_width, canvas_width)
        p_canvas.itemconfig(p_window_id, width=final_width)
        p_canvas.configure(scrollregion=(0, 0, final_width, p_frame.winfo_reqheight()))
        bind_frame_scroll_events(parent, p_canvas, True)

    @property
    def memory_warning_shown(self) -> bool:
        """Check if memory warning has been shown."""
        return self._memory_warning_shown

    @memory_warning_shown.setter
    def memory_warning_shown(self, value: bool) -> None:
        """Set the memory warning shown flag."""
        self._memory_warning_shown = value

    @property
    def flow(self) -> Optional[Type[gui_utils.AbsFlow]]:
        """Get the flow callable."""
        return self._flow

    @flow.setter
    def flow(self, value: Optional[Type[gui_utils.AbsFlow]]) -> None:
        """Set the flow callable and validate its signature."""
        if value is None:
            self._flow = None
            del self.flow_inst
            return
        # Validate that value is an instance of AbsFlow
        assert issubclass(value, gui_utils.AbsFlow), "flow must be an instance of AbsFlow"
        self._flow = value

    @flow.deleter
    def flow(self) -> None:
        """Delete the flow callable."""
        self._flow = None

    @property
    def flow_inst(self) -> Optional[gui_utils.AbsFlow]:
        """Get the flow instance."""
        return self._flow_inst

    @flow_inst.setter
    def flow_inst(self, value: Optional[gui_utils.AbsFlow]) -> None:
        """Set the flow instance."""
        assert isinstance(value, gui_utils.AbsFlow), "flow_inst must be an instance of AbsFlow"
        self._flow_inst = value

    @flow_inst.deleter
    def flow_inst(self) -> None:
        """Delete the flow instance."""
        self._flow_inst = None
        cleanup_vram()

    @property
    def flow_body(self) -> BodyDict:
        """Get the flow body from the current JSON data."""
        if self._flow_body is None:
            raise ValueError("Try to access flow_body when it is not set")
        return self._flow_body

    @flow_body.setter
    def flow_body(self, value: Optional[dict[str, Any]]) -> None:
        """Set the flow body."""
        if not is_bodydict(value):
            raise ValueError("flow_body must conform to BodyDict structure")
        self._flow_body = value

    @flow_body.deleter
    def flow_body(self) -> None:
        """Delete the flow body."""
        self._flow_body = None

    def set_flow_body(self, filename: str) -> None:
        """
        Set the flow body by loading it from a YAML file.

        Args:
            filename (str): Path to the YAML file containing the flow body.
        """
        if filename is None:
            self._flow_body = None
            return

        assert os.path.isfile(filename), f"{filename} is not a valid file"
        # Load YAML file
        with open(filename, "r", encoding="utf-8") as f:
            value: Optional[dict[str, Any]] = yaml.safe_load(f)

        self.flow_body = value

    def __init__(self, root: tk.Tk):
        self._flow = None
        self._flow_body: Optional[BodyDict] = None
        self._has_changes = False
        self.root = root
        self.root.title("JSON Configuration Manager")

        # Get screen dimensions and set window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.current_file: Optional[str] = None
        self._memory_warning_shown = False

        self._setup_ui()
        self._refresh_folder_list()

        # Intercept window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Top controls with horizontal scroll
        controls_container = ttk.Frame(main_frame)
        controls_container.pack(fill="x", pady=(0, 10))

        controls_canvas = tk.Canvas(controls_container, highlightthickness=0, height=35)
        controls_scrollbar = ttk.Scrollbar(controls_container, orient="horizontal", command=controls_canvas.xview)
        controls_frame = ttk.Frame(controls_canvas)

        controls_window_id = controls_canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        controls_canvas.configure(xscrollcommand=controls_scrollbar.set)

        def controls_scroll_action(
            e: tk.Event, p=controls_container, c=controls_canvas, f=controls_frame, w=controls_window_id
        ) -> None:
            """Update scrollregion and frame width when content or canvas changes."""
            self._update_actions_scrollregion(p, c, f, w, e)

        controls_frame.bind("<Configure>", controls_scroll_action)
        controls_canvas.bind("<Configure>", controls_scroll_action)

        controls_scrollbar.pack(side="bottom", fill="x")
        controls_canvas.pack(side="top", fill="x", expand=True)

        # Flow selector
        ttk.Label(controls_frame, text="Flow Folder:").pack(side="left", padx=(0, 5))
        self.folder_var = tk.StringVar()
        self.folder_combo = ttk.Combobox(controls_frame, textvariable=self.folder_var, width=50, state="readonly")
        self.folder_combo.pack(side="left", padx=(0, 10))
        self.folder_combo.bind("<<ComboboxSelected>>", self._on_folder_selected)

        ttk.Button(controls_frame, text="Refresh Flows", command=self._refresh_folder_list).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Show Body", command=self._show_body).pack(side="left", padx=5)

        # File selector
        ttk.Label(controls_frame, text="JSON File:").pack(side="left", padx=(0, 5))
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(controls_frame, textvariable=self.file_var, width=50, state="readonly")
        self.file_combo.pack(side="left", padx=(0, 10))
        self.file_combo.bind("<<ComboboxSelected>>", self._on_file_selected)

        ttk.Button(controls_frame, text="Refresh JSONs", command=self._refresh_file_list).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Check Memory", command=lambda: show_memory_details(self)).pack(
            side="left", padx=5
        )

        # Paned window for editor and images
        paned = ttk.PanedWindow(main_frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left side - JSON Editor
        editor_frame = ttk.LabelFrame(paned, text="JSON Editor", padding="5")
        self.json_editor = JSONTreeEditor(
            editor_frame, on_change=self._mark_changes, on_refresh=self._check_unsaved_changes
        )
        self.json_editor.pack(fill="both", expand=True)
        paned.add(editor_frame, weight=1)

        # Right side - Image Viewer
        viewer_frame = ttk.LabelFrame(paned, text="Output Images", padding="5")

        # Adding a frame for action buttons at the bottom of viewer_frame
        actions_container = ttk.Frame(viewer_frame)
        actions_container.pack(side="bottom", fill="x", pady=2)

        actions_canvas = tk.Canvas(actions_container, highlightthickness=0, height=35)
        actions_scrollbar = ttk.Scrollbar(actions_container, orient="horizontal", command=actions_canvas.xview)
        actions_frame = ttk.Frame(actions_canvas)

        actions_window_id = actions_canvas.create_window((0, 0), window=actions_frame, anchor="nw")
        actions_canvas.configure(xscrollcommand=actions_scrollbar.set)

        def actions_scroll_action(
            e: tk.Event, p=actions_container, c=actions_canvas, f=actions_frame, w=actions_window_id
        ) -> None:
            """Update scrollregion and frame width when content or canvas changes."""
            self._update_actions_scrollregion(p, c, f, w, e)

        actions_frame.bind("<Configure>", actions_scroll_action)
        actions_canvas.bind("<Configure>", actions_scroll_action)

        actions_scrollbar.pack(side="bottom", fill="x")
        actions_canvas.pack(side="top", fill="x", expand=True)

        ttk.Button(actions_frame, text="Save", command=self._save_file).pack(side="left", padx=5)
        ttk.Button(actions_frame, text="Save As", command=self._save_as_file).pack(side="left", padx=5)
        ttk.Button(
            actions_frame,
            text="Clean output Folder",
            command=self._clean_output_folder,
        ).pack(side="left", padx=5)

        # Steps input
        def _validate_steps(new_value: str) -> bool:
            """Validate steps input to be between 1 and 999."""
            if new_value == "":
                return True
            try:
                val = int(new_value)
                return 1 <= val <= 999
            except ValueError:
                return False

        ttk.Button(actions_frame, text="Execute", command=self._execute).pack(side="right", padx=5)
        ttk.Button(
            actions_frame, text="Clean Memory", command=cast(Callable[..., None], lambda ins=self: manual_cleanup(ins))
        ).pack(side="right", padx=5)
        self.multiprocess_var = tk.BooleanVar(value=True)
        multiprocess_check = ttk.Checkbutton(actions_frame, variable=self.multiprocess_var)
        multiprocess_check.pack(side="right", padx=(0, 5))
        ttk.Label(actions_frame, text="Multiprocess:").pack(side="right", padx=(5, 5))
        self.steps_var = tk.IntVar(value=20)
        validate_cmd = (self.root.register(_validate_steps), "%P")
        steps_entry = ttk.Spinbox(
            actions_frame,
            from_=1,
            to=999,
            increment=1,
            wrap=True,
            width=10,
            validate="key",
            validatecommand=validate_cmd,
        )
        steps_entry.pack(side="right", padx=(0, 5))
        bind_frame_scroll_events(steps_entry, steps_entry)
        ttk.Label(actions_frame, text="Steps:").pack(side="right", padx=(20, 5))

        def steps_trace_callback(*_args: Any) -> None:
            """Ensure steps_var is always between 1 and 999."""
            try:
                self.steps_var.get()
                self.steps_var.set(max(1, min(999, self.steps_var.get())))
            except Exception:
                logging.exception("Invalid steps value")
                steps_entry.set(1)

        self.steps_var.trace_add("read", steps_trace_callback)
        steps_entry.config(textvariable=self.steps_var)
        self.image_viewer = ImageViewer(viewer_frame)
        self.image_viewer.pack(side="top", fill="both", expand=True)
        paned.add(viewer_frame, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(fill="x", pady=(10, 0))

    def _mark_changes(self, toggle_changes: bool) -> None:
        """Mark that there are unsaved changes."""
        if toggle_changes:
            if not self._has_changes:
                self._has_changes = True
                self._update_title()
        else:
            self._has_changes = False
            self._update_title()

    def _update_title(self) -> None:
        """Update window title to reflect unsaved changes."""
        base_title = "JSON Configuration Manager"
        if self._has_changes:
            self.root.title(f"*{base_title} - Unsaved Changes")
        else:
            self.root.title(base_title)

    def _check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt user.

        Returns:
            True if it's safe to proceed (no changes or user chose to discard/save).
            False if user cancelled the operation.
        """
        if not self._has_changes:
            return True

        response = messagebox.askyesnocancel(
            "Unsaved Changes",
            (
                "You have unsaved changes.\n\nDo you want to save before continuing?\n\n"
                "If you choose 'No', changes will be discarded."
            ),
            icon="warning",
        )

        if response is None:  # Cancel
            return False
        elif response:  # Yes - save
            self._save_file()
            return not self._has_changes  # Return True only if save succeeded
        else:  # No - discard
            self._on_file_selected(skip_check=True)
            self._mark_changes(False)
            return True

    def _on_close(self) -> None:
        """Handle window close event."""
        if self._check_unsaved_changes():
            # Final cleanup before closing
            cleanup_vram()
            self.root.destroy()

    def _refresh_folder_list(self) -> None:
        """Refresh the list of Flow folders."""
        try:
            folders = [
                f
                for f in os.listdir(gui_utils.get_scripts_folder_path())
                if os.path.isdir(os.path.join(gui_utils.get_scripts_folder_path(), f)) and not f.startswith(("__", "."))
            ]
            folders.sort()
            self.folder_combo["values"] = folders
            self.status_var.set(f"Found {len(folders)} Flow folders")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read directory: {e}")
            logging.exception("Failed to read Flow folders")

    def _show_body(self) -> None:
        """Show the body.yml content."""
        foldername = self.folder_var.get()
        if not foldername:
            messagebox.showwarning("Warning", "No Flow folder selected")
            return

        try:
            _, body_path = gui_utils.get_flow_and_body_paths(foldername)
            if not os.path.exists(body_path):
                messagebox.showerror("Error", f"Body file not found: {body_path}")
                return

            with open(body_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Create window
            win = tk.Toplevel(self.root)
            win.title(f"Body: {os.path.basename(body_path)}")
            win.geometry("600x800")

            # Text widget with scrollbar
            frame = ttk.Frame(win)
            frame.pack(fill="both", expand=True)

            text = tk.Text(frame, wrap="none", font=("Consolas", 10))
            text.insert("1.0", content)
            text.config(state="disabled")

            def select_all(_event=None) -> str:
                """Select all text in the text widget."""
                text.tag_add("sel", "1.0", "end")
                return "break"

            text.bind("<Control-a>", select_all)

            v_scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            h_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)

            text.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

            v_scroll.pack(side="right", fill="y")
            h_scroll.pack(side="bottom", fill="x")
            text.pack(side="left", fill="both", expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load body file:\n{e}")
            logging.exception("Failed to show body file")

    def _refresh_file_list(self) -> None:
        """Refresh the list of JSON files."""
        foldername = self.folder_var.get()
        assert foldername, "Folder name is empty"
        try:
            look_path = os.path.join(gui_utils.get_main_images_path(), foldername)
            logging.debug("Refreshing JSON file list in folder: %s", look_path)
            files = [f for f in os.listdir(look_path) if f.endswith(".json")]
            files.sort()
            self.file_combo["values"] = files
            self.status_var.set(f"Found {len(files)} JSON files")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read directory: {e}")
            logging.exception("Failed to read JSON files in folder %s", foldername)

    def _on_folder_selected(self, _event: tk.Event | None = None) -> None:
        """Handle file selection."""
        try:
            foldername = self.folder_var.get()
            if not foldername:
                return

            if not self._check_unsaved_changes():
                return

            filepath = os.path.join(gui_utils.get_main_images_path(), foldername)
            logging.debug("Selected Flow folder: %s", filepath)

            # if not exists, create it
            if not os.path.exists(filepath):
                os.makedirs(filepath)

            # validate that foldername is a directory
            assert os.path.isdir(filepath), f"{foldername} is not a valid directory"
            del self.flow
            flow, body = gui_utils.get_flow_and_body_paths(foldername)

            def load_script() -> None:
                """Load the script for the selected folder and set the flow function."""
                # verify that the script has a main function
                module_path = Path(flow)
                spec = spec_from_file_location(module_path.stem, flow)
                if not spec:
                    raise ModuleNotFoundError(f"Could not load spec for module {module_path.stem}")
                module = module_from_spec(spec)
                if not spec.loader:
                    raise ModuleNotFoundError(f"Could not load module {module_path.stem}")
                spec.loader.exec_module(module)
                assert hasattr(module, "Flow"), f"Script {flow} does not have a main function"
                self.flow = getattr(module, "Flow")

            loading_modal.show_loading_modal(self.root, load_script, (), f"Loading Flow: {foldername}...")
            assert self.flow is not None, "Flow function is not set after loading script"

            # Set flow body
            self.set_flow_body(body)

            # Clear previous data
            self.json_editor.load_data({}, {"props": {}})
            self.current_file = None
            self._mark_changes(False)
            self.status_var.set(f"Selected folder: {foldername}")
            self.image_viewer.clear()
            self._refresh_file_list()
        except ModuleNotFoundError:
            messagebox.showerror("Error", f"Script not found for folder: {foldername}")
            logging.exception("Script not found for folder %s", foldername)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {foldername},\n{e}")
            logging.exception("Failed to load folder %s", foldername)

    def _on_file_selected(self, _event: tk.Event | None = None, skip_check: bool = False) -> None:
        """Handle file selection."""
        foldername = self.folder_var.get()
        assert foldername, "Folder name is empty"
        flow_body = self.flow_body
        assert flow_body is not None, "Flow body is not set"
        filename = self.file_var.get()
        if not filename:
            return

        if not skip_check and not self._check_unsaved_changes():
            return

        body = self.flow_body
        assert body is not None, "Flow body is not set"

        filepath = os.path.join(gui_utils.get_main_images_path(), foldername, filename)
        filename_without_ext = os.path.splitext(filename)[0]

        def load_file() -> None:
            """Load the selected JSON file and execute the flow."""
            try:
                logging.info("Loading JSON file: %s", filepath)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.json_editor.load_data(data, body)
                self.current_file = filepath
                self._mark_changes(False)
                self.status_var.set(f"Loaded: {filename}")
                self.image_viewer.clear()
                del self.flow_inst
                # Check memory status before execution
                if not check_memory_available():
                    raise MemoryError("GPU memory is too fragmented. Please restart the application.")
                logging.info("Creating Flow instance for file: %s", filepath)
                assert self.flow is not None, "Flow class is not set"
                assert issubclass(self.flow, gui_utils.AbsFlow), "Flow must be a subclass of AbsFlow"
                # Execute the flow
                self.flow_inst = self.flow(foldername, filename_without_ext)  # pylint: disable=E1102
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON Error", f"Failed to parse JSON:\n{e}")
                logging.exception("Failed to parse JSON file %s", filepath)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{e}")
                logging.exception("Failed to load file %s", filepath)

        loading_modal.show_loading_modal(
            self.root,
            load_file,
            (),
            f"Loading Flow {foldername}: {filename_without_ext}...",
            False,
            p_logger.poll_log_queue,
        )

    def _save_file(self) -> None:
        """Save changes to the current file."""
        if not self.current_file:
            messagebox.showwarning("Warning", "No file selected")
            return

        try:
            data = self.json_editor.get_data()
            with open(self.current_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._mark_changes(False)
            self.status_var.set(f"Saved: {os.path.basename(self.current_file)}")
            loading_modal.auto_close_info(self.root, "Success", "File saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            logging.exception("Failed to save file %s", self.current_file)

    def _save_as_file(self) -> None:
        """Save as a new file with user provided name or UUID."""
        try:
            data = self.json_editor.get_data()

            filename = simpledialog.askstring("Save As", "Enter filename (leave blank for UUID):", parent=self.root)
            if filename is None:
                return

            filename = filename.strip()

            if not filename:
                new_uuid = str(uuid.uuid4())
                new_filename = f"{new_uuid}.json"
            else:
                if " " in filename:
                    messagebox.showerror("Error", "Spaces are not allowed in filename")
                    return
                new_filename = filename if filename.endswith(".json") else f"{filename}.json"
            foldername = self.folder_var.get()
            assert foldername, "Folder name is empty"
            new_filepath = os.path.join(gui_utils.get_main_images_path(), foldername, new_filename)

            with open(new_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._refresh_file_list()
            self.file_var.set(new_filename)
            self.current_file = new_filepath
            self._mark_changes(False)
            self.status_var.set(f"Saved as: {new_filename}")
            messagebox.showinfo("Success", f"File saved as:\n{new_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            logging.exception("Failed to save file as new file")

    def _clean_output_folder(self) -> None:
        """Clean the output folder (temp directory)."""
        output_dir = folder_paths.get_output_directory()

        if not os.path.exists(output_dir):
            return

        if messagebox.askyesno("Confirm", f"Are you sure you want to delete all files in {output_dir}?"):
            try:
                for f in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, f)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                self.image_viewer.clear()
                self.status_var.set("Output folder cleaned")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clean folder: {e}")

    def _execute(self) -> None:
        """Execute the main function with the selected JSON."""
        if not self.current_file:
            messagebox.showwarning("Warning", "No file selected")
            return

        if not self._check_unsaved_changes():
            return

        try:
            steps = self.steps_var.get()
        except ValueError:
            messagebox.showerror("Error", "Steps must be a number")
            return
        multiprocess = self.multiprocess_var.get()

        # Get filename without extension
        foldername = self.folder_var.get()
        try:
            assert foldername, "Folder name is empty"
            assert foldername in self.current_file, "Folder name must be part of the current file path"
        except AssertionError as e:
            messagebox.showerror("Error", f"Failed to execute:\n{e}")
            logging.exception("Execution failed")
            raise e

        inst = self.flow_inst
        if inst is None:
            raise RuntimeError("Flow instance is not set")

        self.status_var.set(f"Executing with {inst.file_path}...")
        self.root.update()

        def run_flow_direct() -> None:
            """Run the flow directly in this process."""
            try:
                inst = self.flow_inst

                assert inst is not None, "Flow instance is not set"

                logging.info(
                    "Executing flow: script=%s, folder=%s, file=%s, steps=%s",
                    foldername,
                    foldername,
                    inst.file_path,
                    steps,
                )

                image_paths = inst.run(steps, multiprocess)
                # Clean up after execution
                cleanup_vram()

                # Check memory after execution and warn if fragmented
                check_memory_fragmentation(self)

                # Display results
                if image_paths:
                    self.image_viewer.display_images(image_paths)
                    self.status_var.set(f"Execution complete. Generated {len(image_paths)} images.")
                else:
                    self.status_var.set("Execution complete. No images generated.")
                    messagebox.showinfo("Info", "Execution completed but no images were generated.")

            except MemoryError as e:
                error_msg = str(e)
                logging.error("Memory error: %s", error_msg)
                self.root.after(
                    0,
                    cast(
                        Callable[..., str],
                        lambda msg=error_msg: messagebox.showerror(
                            "Memory Error", f"{msg}\n\nPlease close and reopen the application."
                        ),
                    ),
                )
                raise

        loading_modal.show_loading_modal(
            self.root,
            run_flow_direct,
            (),
            f"Executing Flow {foldername}: {inst.file_path}...",
            True,
            p_logger.poll_log_queue,
        )


def main() -> None:
    """Main entry point."""
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    JSONManagerApp(root)
    root.mainloop()
