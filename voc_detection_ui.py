"""Interactive VOC object-detection UI for showing top-3 YOLO results."""

from pathlib import Path
import argparse
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

from detection_cnn import _require_ultralytics


class VOCDetectionApp:
    """Small tkinter UI: image with boxes on the left, top-3 detections on the right."""

    def __init__(self, root, weights, image_dir, conf=0.25, imgsz=640, device=None):
        self.root = root
        self.root.title("VOC YOLO detection demo")
        self.weights = Path(weights)
        self.image_dir = Path(image_dir)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.index = 0
        self.current_photo = None

        if not self.weights.exists():
            raise FileNotFoundError(f"Cannot find weights: {self.weights}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Cannot find image folder: {self.image_dir}")

        self.image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

        YOLO = _require_ultralytics()
        self.model = YOLO(str(self.weights))
        self._build_ui()
        self.detect_current()

    def _build_ui(self):
        outer = tk.Frame(self.root, bg="#111111", padx=16, pady=16)
        outer.pack(fill=tk.BOTH, expand=True)

        main = tk.Frame(outer, bg="#111111")
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, bg="#111111")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 18))

        self.image_label = tk.Label(left, bg="#050505")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(left, bg="#111111")
        controls.pack(fill=tk.X, pady=(12, 0))

        self.prev_btn = tk.Button(controls, text="Prev", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT)

        self.detect_btn = tk.Button(controls, text="Detect", command=self.detect_current)
        self.detect_btn.pack(side=tk.LEFT, padx=8)

        self.next_btn = tk.Button(controls, text="Next", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT)

        self.file_label = tk.Label(
            controls,
            text="",
            fg="white",
            bg="#111111",
            anchor="e",
        )
        self.file_label.pack(side=tk.RIGHT)

        right = tk.Frame(main, bg="#111111", width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(
            right,
            text="Top 3 detections",
            fg="white",
            bg="#111111",
            font=("Arial", 18, "bold"),
        ).pack(anchor="w", pady=(0, 12))

        self.result_canvas = tk.Canvas(
            right,
            width=320,
            height=260,
            bg="#111111",
            highlightthickness=0,
        )
        self.result_canvas.pack(fill=tk.X)

        self.summary_label = tk.Label(
            right,
            text="",
            fg="#dddddd",
            bg="#111111",
            justify=tk.LEFT,
            anchor="nw",
            font=("Arial", 11),
            wraplength=300,
        )
        self.summary_label.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def prev_image(self):
        self.index = (self.index - 1) % len(self.image_paths)
        self.detect_current()

    def next_image(self):
        self.index = (self.index + 1) % len(self.image_paths)
        self.detect_current()

    def detect_current(self):
        path = self.image_paths[self.index]
        self.file_label.config(text=f"{self.index + 1}/{len(self.image_paths)}  {path.name}")
        self.root.update_idletasks()

        results = self.model.predict(
            source=str(path),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            save=False,
            verbose=False,
        )
        result = results[0]
        plotted_bgr = result.plot()
        plotted_rgb = plotted_bgr[..., ::-1]
        self._show_image(plotted_rgb)

        detections = self._extract_top3(result)
        self._show_top3(detections)

    def _show_image(self, rgb_array):
        image = Image.fromarray(rgb_array)
        max_w, max_h = 820, 620
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.current_photo)

    def _extract_top3(self, result):
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            score = float(box.conf.item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append(
                {
                    "class": self.model.names[cls_id],
                    "score": score,
                    "xyxy": xyxy,
                }
            )
        detections.sort(key=lambda item: item["score"], reverse=True)
        return detections[:3]

    def _show_top3(self, detections):
        self.result_canvas.delete("all")
        if not detections:
            self.result_canvas.create_text(
                10,
                20,
                anchor="nw",
                text="No detection above threshold",
                fill="white",
                font=("Arial", 13),
            )
            self.summary_label.config(text="Try a lower confidence threshold or another image.")
            return

        canvas_w = int(self.result_canvas.winfo_width() or 320)
        bar_x0 = 18
        bar_w = canvas_w - 36
        y = 28
        lines = []

        for rank, item in enumerate(detections, 1):
            score = item["score"]
            label = f"{rank}. {item['class']}  {score:.1%}"
            color = "#e53935" if rank == 1 else "#9e9e9e"
            self.result_canvas.create_text(
                bar_x0,
                y - 18,
                anchor="nw",
                text=label,
                fill="white",
                font=("Arial", 12, "bold" if rank == 1 else "normal"),
            )
            self.result_canvas.create_rectangle(
                bar_x0,
                y,
                bar_x0 + bar_w,
                y + 18,
                fill="#2a2a2a",
                outline="",
            )
            self.result_canvas.create_rectangle(
                bar_x0,
                y,
                bar_x0 + bar_w * score,
                y + 18,
                fill=color,
                outline="",
            )
            lines.append(f"{rank}. {item['class']}: {score:.1%}, box={item['xyxy']}")
            y += 78

        self.summary_label.config(text="\n".join(lines))


def default_image_dir():
    base = Path(__file__).resolve().parent
    candidates = [
        base / "VOCdevkit" / "VOC2007test" / "JPEGImages",
        base / "VOCdevkit" / "VOC2007" / "JPEGImages",
        base / "VOCdevkit" / "VOC2007trainval" / "JPEGImages",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="VOC YOLO detection UI")
    parser.add_argument(
        "--weights",
        default=str(base / "outputs" / "voc2007_yolo" / "runs" / "yolo_voc2007" / "weights" / "best.pt"),
    )
    parser.add_argument("--image-dir", default=str(default_image_dir()))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = tk.Tk()
    try:
        VOCDetectionApp(
            root,
            weights=args.weights,
            image_dir=args.image_dir,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
        )
    except Exception as exc:
        messagebox.showerror("VOC detection UI", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
