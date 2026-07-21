#!/usr/bin/env python3
"""
gui.py — A small Tkinter front-end for process.py.

Works on a whole folder: it lists every image, lets you click or arrow through
them, and previews the current settings on the selected one. "Process whole
folder →" applies those settings to every image at once, with a progress bar and
the same skip-existing / overwrite behaviour as the CLI.

Shows a live BEFORE / AFTER preview at the top and all of the processing
settings (magic wand, iPhone-style sliders, rotation, resize, quality)
underneath as buttons and sliders. Moving a slider re-renders the "after"
preview live. "Save this image…" writes the single selected image.

The image maths is NOT duplicated here: the GUI imports process.py and calls
process.transform_image() with a process.default_args() namespace, so the CLI
and the GUI always produce identical output.

    python gui.py                 # opens the editor and creates ./in and ./out if needed
    python gui.py path/to/img.webp   # opens that image's folder, selecting it
    python gui.py path/to/folder     # opens a specific folder

The command line tool (process.py) still works exactly as before; this is an
optional companion, not a replacement.

Dependencies:
    pip install Pillow numpy    (Tkinter ships with Python)
"""

from __future__ import annotations

import random
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import process  # single source of truth for the image pipeline

PREVIEW_MAX = 300          # px, size of each preview pane
CHECKER = (235, 235, 235), (200, 200, 200)   # transparency checkerboard colours

# Slider spec: (attr name on args, label, min, max, default, resolution).
SLIDERS = [
    ("exposure",       "Exposure",     -100, 100,   0, 1),
    ("brightness",     "Brightness",   -100, 100,   0, 1),
    ("contrast",       "Contrast",     -100, 100,   0, 1),
    ("highlights",     "Highlights",   -100, 100,   0, 1),
    ("shadows",        "Shadows",      -100, 100,   0, 1),
    ("black_point",    "Black Point",  -100, 100,   0, 1),
    ("saturation_adj", "Saturation",   -100, 100,   0, 1),
    ("vibrance",       "Vibrance",     -100, 100,   0, 1),
    ("warmth",         "Warmth",       -100, 100,   0, 1),
    ("tint",           "Tint",         -100, 100,   0, 1),
    ("sharpness",      "Sharpness",    -100, 100,   0, 1),
    ("vignette",       "Vignette",     -100, 100,   0, 1),
]


def ensure_directory(path: str | Path) -> Path:
    """Create a directory (and any missing parents) relative to the current execution folder."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def checkerboard(size: tuple[int, int], square: int = 12) -> Image.Image:
    """An opaque RGBA checkerboard, so transparency in the preview is visible."""
    w, h = size
    bg = Image.new("RGBA", size, CHECKER[0] + (255,))
    dark = Image.new("RGBA", (square, square), CHECKER[1] + (255,))
    for y in range(0, h, square):
        for x in range(0, w, square):
            if (x // square + y // square) % 2:
                bg.paste(dark, (x, y))
    return bg


def fit_preview(im: Image.Image, box: int = PREVIEW_MAX) -> Image.Image:
    """Downscale (never upscale) onto a checkerboard for display."""
    im = im.convert("RGBA")
    thumb = im.copy()
    thumb.thumbnail((box, box), Image.LANCZOS)
    board = checkerboard(thumb.size)
    board.alpha_composite(thumb)
    return board


class EditorApp:
    def __init__(self, root: tk.Tk, initial: Path | None = None):
        self.root = root
        root.title("Batch image processor — process.py front-end")
        root.minsize(760, 560)
        root.geometry("1080x620")

        self.src_path: Path | None = None
        self.src_image: Image.Image | None = None   # full-res original (RGBA)
        self.after_image: Image.Image | None = None  # last rendered result (full-res)
        self._before_tk = None                       # keep refs so Tk doesn't GC them
        self._after_tk = None
        self._render_job = None                      # debounce handle

        self.folder: Path | None = None              # currently loaded folder
        self.file_list: list[Path] = []              # images in that folder
        self.index: int = -1                         # selected image in file_list

        # args namespace mirrors the CLI defaults; sliders mutate it in place.
        self.args = process.default_args()

        self._build_ui()

        # Startup: create the default working folders in the current execution directory,
        # then load the default input folder if it exists.
        ensure_directory(Path.cwd() / "in")
        ensure_directory(Path.cwd() / "out")
        if initial and initial.is_file():
            self.load_folder(initial.parent, select=initial)
        elif initial and initial.is_dir():
            self.load_folder(initial)
        else:
            default_dir = Path.cwd() / "in"
            if default_dir.is_dir():
                self.load_folder(default_dir)

    # ---- UI construction -------------------------------------------------- #
    def _build_ui(self) -> None:
        # Top area: a file list on the left, the before/after previews on the right.
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        self._build_filelist(top)

        previews = ttk.Frame(top)
        previews.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.before_lbl = self._preview_pane(previews, "Before", 0)
        self.after_lbl = self._preview_pane(previews, "After", 1)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.rowconfigure(0, weight=1)

        # Controls area underneath.
        controls = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        controls.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(
            controls,
            text="Batch workflow: drop source images into ./in, then process them to ./out.",
            foreground="#2f5d7a",
            wraplength=980,
            justify=tk.LEFT,
        ).pack(side=tk.TOP, anchor=tk.W, pady=(0, 6))
        self._build_toolbar(controls)
        self._build_sliders(controls)
        self._build_geometry(controls)
        self._build_actions(controls)

        # Status line.
        self.status = tk.StringVar(value="Open a folder or image to begin.")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W, padding=4).pack(side=tk.BOTTOM, fill=tk.X)

        # Keyboard navigation across the folder (PageUp/PageDown + Ctrl+arrows).
        self.root.bind("<Prior>", lambda _e: self.select_relative(-1))
        self.root.bind("<Next>", lambda _e: self.select_relative(+1))
        self.root.bind("<Control-Left>", lambda _e: self.select_relative(-1))
        self.root.bind("<Control-Right>", lambda _e: self.select_relative(+1))

    def _build_filelist(self, parent):
        frame = ttk.LabelFrame(parent, text="Folder", padding=6)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

        nav = ttk.Frame(frame)
        nav.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(nav, text="◀", width=3,
                   command=lambda: self.select_relative(-1)).pack(side=tk.LEFT)
        ttk.Button(nav, text="▶", width=3,
                   command=lambda: self.select_relative(+1)).pack(side=tk.LEFT, padx=(2, 0))
        self.count_lbl = ttk.Label(nav, text="0 / 0", anchor=tk.CENTER)
        self.count_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        list_wrap = ttk.Frame(frame)
        list_wrap.pack(side=tk.TOP, fill=tk.Y, expand=True, pady=(4, 0))
        sb = ttk.Scrollbar(list_wrap, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(list_wrap, width=26, height=12,
                                  activestyle="dotbox",
                                  yscrollcommand=sb.set, exportselection=False)
        sb.config(command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

    def _preview_pane(self, parent, title, col):
        frame = ttk.LabelFrame(parent, text=title, padding=6)
        frame.grid(row=0, column=col, padx=6, sticky="nsew")
        lbl = ttk.Label(frame, anchor=tk.CENTER, width=PREVIEW_MAX // 8)
        lbl.pack(fill=tk.BOTH, expand=True)
        return lbl

    def _build_toolbar(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(side=tk.TOP, fill=tk.X, pady=(2, 4))

        ttk.Button(bar, text="Open folder…", command=self.on_open_folder).pack(side=tk.LEFT)
        ttk.Button(bar, text="Open image…", command=self.on_open).pack(side=tk.LEFT, padx=(6, 0))

        self.magic_var = tk.BooleanVar(value=self.args.magic_wand)
        ttk.Checkbutton(bar, text="✨ Magic wand (auto-optimise)",
                        variable=self.magic_var,
                        command=self._on_change).pack(side=tk.LEFT, padx=12)

        ttk.Button(bar, text="Reset all",
                   command=self.reset_all).pack(side=tk.RIGHT)

    def _build_sliders(self, parent):
        box = ttk.LabelFrame(parent, text="Adjustments  (−100 … +100, 0 = off)",
                             padding=6)
        box.pack(side=tk.TOP, fill=tk.X, pady=4)

        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.value_lbls: dict[str, ttk.Label] = {}

        # Two columns of sliders to keep the window compact.
        left = ttk.Frame(box); left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.Frame(box); right.grid(row=0, column=1, sticky="nsew")
        box.columnconfigure(0, weight=1)
        box.columnconfigure(1, weight=1)

        half = (len(SLIDERS) + 1) // 2
        for i, (attr, label, lo, hi, dflt, res) in enumerate(SLIDERS):
            parent_col = left if i < half else right
            self._one_slider(parent_col, attr, label, lo, hi, dflt, res)

    def _one_slider(self, parent, attr, label, lo, hi, dflt, res):
        row = ttk.Frame(parent)
        row.pack(side=tk.TOP, fill=tk.X, pady=2)

        ttk.Label(row, text=label, width=11, anchor=tk.W).pack(side=tk.LEFT)

        var = tk.DoubleVar(value=getattr(self.args, attr, dflt))
        self.slider_vars[attr] = var

        val_lbl = ttk.Label(row, width=5, anchor=tk.E)
        val_lbl.pack(side=tk.RIGHT)
        self.value_lbls[attr] = val_lbl

        ttk.Button(row, text="⟲", width=2,
                   command=lambda a=attr, d=dflt: self.reset_slider(a, d)
                   ).pack(side=tk.RIGHT, padx=(4, 4))

        scale = ttk.Scale(row, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL,
                          command=lambda _v, a=attr: self._on_slider(a))
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self._update_value_label(attr)

    def _build_geometry(self, parent):
        box = ttk.LabelFrame(parent, text="Rotation, size & quality", padding=6)
        box.pack(side=tk.TOP, fill=tk.X, pady=4)

        # Rotation mode: off / random / fixed angle.
        ttk.Label(box, text="Rotate:").grid(row=0, column=0, sticky=tk.W)
        self.rotate_mode = tk.StringVar(value="off")
        for i, mode in enumerate(("off", "random", "fixed")):
            ttk.Radiobutton(box, text=mode.capitalize(), value=mode,
                            variable=self.rotate_mode,
                            command=self._on_change).grid(row=0, column=1 + i,
                                                          sticky=tk.W, padx=4)
        self.angle_var = tk.DoubleVar(value=0)
        self.angle_scale = ttk.Scale(box, from_=0, to=360, variable=self.angle_var,
                                     orient=tk.HORIZONTAL,
                                     command=lambda _v: self._on_change())
        self.angle_scale.grid(row=0, column=4, sticky="ew", padx=6)
        self.angle_lbl = ttk.Label(box, width=5, text="0°")
        self.angle_lbl.grid(row=0, column=5)

        # Max size.
        ttk.Label(box, text="Max size (px, 0 = none):").grid(row=1, column=0,
                                                             columnspan=2, sticky=tk.W,
                                                             pady=(6, 0))
        self.maxsize_var = tk.IntVar(value=self.args.max_size)
        ttk.Spinbox(box, from_=0, to=4096, increment=16, width=7,
                    textvariable=self.maxsize_var,
                    command=self._on_change).grid(row=1, column=2, sticky=tk.W,
                                                  pady=(6, 0))

        # Quality.
        ttk.Label(box, text="WebP quality:").grid(row=1, column=3, sticky=tk.E,
                                                  pady=(6, 0))
        self.quality_var = tk.IntVar(value=self.args.quality)
        ttk.Spinbox(box, from_=1, to=100, width=5, textvariable=self.quality_var,
                    command=self._on_change).grid(row=1, column=4, sticky=tk.W,
                                                  pady=(6, 0))
        box.columnconfigure(4, weight=1)

    def _build_actions(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(side=tk.TOP, fill=tk.X, pady=(8, 2))
        ttk.Button(bar, text="Save this image…", command=self.on_save).pack(side=tk.LEFT)
        self.process_btn = ttk.Button(bar, text="Process whole folder →",
                                      command=self.on_batch)
        self.process_btn.pack(side=tk.LEFT, padx=8)

        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Overwrite existing",
                        variable=self.overwrite_var).pack(side=tk.LEFT)

        ttk.Button(bar, text="Copy CLI command",
                   command=self.copy_cli).pack(side=tk.LEFT, padx=8)
        ttk.Button(bar, text="Quit", command=self.root.destroy).pack(side=tk.RIGHT)

        # Progress bar for batch runs (hidden until used).
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

    # ---- events ----------------------------------------------------------- #
    def _on_slider(self, attr):
        self._update_value_label(attr)
        self._on_change()

    def _update_value_label(self, attr):
        self.value_lbls[attr].config(text=f"{self.slider_vars[attr].get():+.0f}")

    def reset_slider(self, attr, dflt):
        self.slider_vars[attr].set(dflt)
        self._update_value_label(attr)
        self._on_change()

    def reset_all(self):
        for attr, _l, _lo, _hi, dflt, _r in SLIDERS:
            self.slider_vars[attr].set(dflt)
            self._update_value_label(attr)
        self.magic_var.set(False)
        self.rotate_mode.set("off")
        self.angle_var.set(0)
        self.maxsize_var.set(512)
        self.quality_var.set(90)
        self._on_change()

    def _on_change(self, *_):
        """Any control changed → sync args and schedule a debounced re-render."""
        self.angle_lbl.config(text=f"{self.angle_var.get():.0f}°")
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(60, self.render_after)

    # ---- args syncing ----------------------------------------------------- #
    def _sync_args(self):
        for attr, *_ in ((s[0],) for s in SLIDERS):
            setattr(self.args, attr, float(self.slider_vars[attr].get()))
        self.args.magic_wand = bool(self.magic_var.get())
        self.args.max_size = int(self.maxsize_var.get())
        self.args.quality = int(self.quality_var.get())
        # Rotation mode maps onto the CLI's --rotate string.
        mode = self.rotate_mode.get()
        if mode == "off":
            self.args.rotate = "off"
        elif mode == "random":
            self.args.rotate = "random"
        else:
            self.args.rotate = str(self.angle_var.get())

    def _current_angle(self, rng: random.Random) -> float:
        return process.resolve_angle(self.args.rotate, rng)

    # ---- folder handling & navigation ------------------------------------ #
    def load_folder(self, folder: Path, select: Path | None = None):
        """Scan a folder for images, populate the list, and show the first."""
        folder = Path(folder)
        images = sorted(p for p in folder.iterdir()
                        if p.is_file() and p.suffix.lower() in process.SUPPORTED_EXT)
        if not images:
            messagebox.showinfo("Open folder", f"No images found in {folder}")
            return
        self.folder = folder
        self.file_list = images
        self.listbox.delete(0, tk.END)
        for p in images:
            self.listbox.insert(tk.END, p.name)
        # Choose which image to show first.
        start = 0
        if select is not None:
            try:
                start = images.index(Path(select))
            except ValueError:
                start = 0
        self.select_index(start)
        self.status.set(f"Loaded folder {folder}  ({len(images)} images)")

    def select_index(self, i: int):
        """Select image #i in the folder and load it into the preview."""
        if not self.file_list:
            return
        i = max(0, min(i, len(self.file_list) - 1))
        self.index = i
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(i)
        self.listbox.activate(i)
        self.listbox.see(i)
        self.count_lbl.config(text=f"{i + 1} / {len(self.file_list)}")
        self.load_image(self.file_list[i])

    def select_relative(self, delta: int):
        if self.file_list:
            self.select_index(self.index + delta)

    def _on_list_select(self, _event):
        sel = self.listbox.curselection()
        if sel and sel[0] != self.index:
            self.select_index(sel[0])

    # ---- rendering -------------------------------------------------------- #
    def load_image(self, path: Path):
        try:
            im = Image.open(path).convert("RGBA")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Open failed", f"{path}\n\n{exc}")
            return
        self.src_path = Path(path)
        self.src_image = im
        self._before_tk = ImageTk.PhotoImage(fit_preview(im))
        self.before_lbl.config(image=self._before_tk)
        self.status.set(f"{self.src_path.name}   ({im.width}×{im.height})")
        self.render_after()

    def render_after(self):
        self._render_job = None
        if self.src_image is None:
            return
        self._sync_args()
        # Preview uses a fixed angle even in "random" mode so the image doesn't
        # jump on every slider move; batch/save re-rolls per image as the CLI does.
        rng = random.Random(0)
        angle = self._current_angle(rng)
        src = self.src_image

        def work():
            # CPU-bound numpy/Pillow work happens off the UI thread. The
            # transform and the preview downscale are plain PIL (thread-safe);
            # only the Tk PhotoImage must be built on the main thread, so we
            # hand the finished PIL image back via root.after().
            try:
                out = process.transform_image(src, self.args, angle)
                preview = fit_preview(out)
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda e=exc: self.status.set(f"Render error: {e}"))
                return
            self.root.after(0, lambda: self._show_after(out, preview))

        # Render off the UI thread so dragging stays smooth.
        threading.Thread(target=work, daemon=True).start()

    def _show_after(self, out, preview):
        # Runs on the main thread: safe to create the Tk image here.
        self.after_image = out
        self._after_tk = ImageTk.PhotoImage(preview)
        self.after_lbl.config(image=self._after_tk)
        self.status.set(f"Preview {out.width}×{out.height}   |   {self.cli_string()}")

    # ---- output ----------------------------------------------------------- #
    def on_open_folder(self):
        default_in = Path.cwd() / "in"
        path = filedialog.askdirectory(
            title="Open a folder of images",
            initialdir=str(default_in if default_in.exists() else Path.cwd()),
        )
        if path:
            self.load_folder(Path(path))

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.webp *.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All files", "*.*")])
        if path:
            # Load the whole containing folder so navigation still works, and
            # jump straight to the chosen file.
            self.load_folder(Path(path).parent, select=Path(path))

    def on_save(self):
        if self.src_image is None:
            messagebox.showinfo("Nothing to save", "Open an image first.")
            return
        self._sync_args()
        default = (self.src_path.stem + ".webp") if self.src_path else "edited.webp"
        path = filedialog.asksaveasfilename(
            title="Save WebP as", defaultextension=".webp",
            initialfile=default, filetypes=[("WebP", "*.webp")])
        if not path:
            return
        save_path = Path(path)
        ensure_directory(save_path.parent)
        rng = random.Random(self.args.seed if self.args.seed is not None else 0)
        out = process.transform_image(self.src_image, self.args, self._current_angle(rng))
        out.save(save_path, format="WEBP", quality=self.args.quality, method=6)
        self.status.set(f"Saved {save_path}")

    def on_batch(self):
        """Apply the current settings to every image in the loaded folder."""
        # Source: the folder that's already loaded (fall back to asking).
        in_p = self.folder
        if in_p is None:
            picked = filedialog.askdirectory(title="Source folder (images to process)")
            if not picked:
                return
            in_p = Path(picked)

        images = [p for p in self.file_list] if self.folder == in_p else sorted(
            p for p in in_p.iterdir()
            if p.is_file() and p.suffix.lower() in process.SUPPORTED_EXT)
        if not images:
            messagebox.showinfo("Process folder", f"No images found in {in_p}")
            return

        # Destination: default to ./out, let the user confirm/redirect.
        default_out = Path.cwd() / "out"
        out_dir = filedialog.askdirectory(
            title=f"Destination folder for {len(images)} processed images",
            initialdir=str(default_out))
        if not out_dir:
            return
        out_p = Path(out_dir)
        ensure_directory(out_p)

        self._sync_args()
        self.args.in_dir = str(in_p)
        self.args.out_dir = str(out_p)
        self.args.overwrite = bool(self.overwrite_var.get())

        # Run off the UI thread so a big folder doesn't freeze the window.
        self.process_btn.config(state=tk.DISABLED)
        self.progress.config(maximum=len(images), value=0)
        threading.Thread(target=self._run_batch, args=(images, out_p), daemon=True).start()

    def _run_batch(self, images: list[Path], out_p: Path):
        rng = random.Random(self.args.seed)
        ok = fail = skipped = 0
        for n, src in enumerate(images, 1):
            dest = out_p / (src.stem + ".webp")
            angle = self._current_angle(rng)   # re-roll per image (matches CLI)
            if dest.exists() and not self.args.overwrite:
                skipped += 1
            else:
                try:
                    process.process_image(src, dest, self.args, angle)
                    ok += 1
                except Exception:  # noqa: BLE001
                    fail += 1
            # Marshal UI updates back to the main thread.
            self.root.after(0, self._batch_progress, n, ok, skipped, fail)
        self.root.after(0, self._batch_done, out_p, ok, skipped, fail)

    def _batch_progress(self, n, ok, skipped, fail):
        self.progress.config(value=n)
        self.status.set(f"Processing folder… {ok} done, {skipped} skipped, {fail} failed")

    def _batch_done(self, out_p, ok, skipped, fail):
        self.progress.config(value=0)
        self.process_btn.config(state=tk.NORMAL)
        messagebox.showinfo(
            "Folder processed",
            f"{ok} processed, {skipped} skipped (already existed), {fail} failed.\n"
            f"Output: {out_p.resolve()}")
        self.status.set(f"Done: {ok} processed, {skipped} skipped, {fail} failed "
                        f"→ {out_p}")

    def cli_string(self) -> str:
        """The equivalent process.py command for the current settings."""
        parts = ["python process.py"]
        if self.args.magic_wand:
            parts.append("--magic-wand")
        if self.args.in_dir != "in":
            parts.append(f"--in {self.args.in_dir}")
        if self.args.out_dir != "out":
            parts.append(f"--out {self.args.out_dir}")
        for attr, label, *_ in SLIDERS:
            v = getattr(self.args, attr)
            if v:
                parts.append(f"--{attr.replace('_', '-')} {v:g}")
        if self.args.saturation != 1.05:
            parts.append(f"--saturation {self.args.saturation:g}")
        if self.args.rotate != "off":
            parts.append(f"--rotate {self.args.rotate}")
        if self.args.max_size != 512:
            parts.append(f"--max-size {self.args.max_size}")
        if self.args.quality != 90:
            parts.append(f"--quality {self.args.quality}")
        return " ".join(parts)

    def copy_cli(self):
        self._sync_args()
        cmd = self.cli_string()
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)
        self.status.set(f"Copied: {cmd}")


def main() -> int:
    initial = None
    if len(sys.argv) > 1:
        cand = Path(sys.argv[1])
        if cand.is_file():
            initial = cand
    root = tk.Tk()
    # A slightly nicer default theme where available.
    try:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f7fb")
        style.configure("TLabelframe", background="#f4f7fb")
        style.configure("TLabelframe.Label", background="#f4f7fb")
        style.configure("TLabel", background="#f4f7fb")
        style.configure("TButton", padding=(8, 4))
    except tk.TclError:
        pass
    EditorApp(root, initial)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
