"""Interactive 28x28 MNIST drawing UI with real-time probability bars."""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import torch
from PIL import Image, ImageDraw

from mnist_cnn import LeNetMNIST


class MNISTDrawApp:
    def __init__(self, root, checkpoint_path, canvas_scale=10):
        self.root = root
        self.root.title("MNIST CNN realtime demo")
        self.checkpoint_path = Path(checkpoint_path)
        self.canvas_scale = canvas_scale
        self.grid_size = 28
        self.canvas_size = self.grid_size * self.canvas_scale
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.brush_size = tk.IntVar(value=2)
        self.after_id = None
        self.last_x = None
        self.last_y = None

        self.image = Image.new("L", (self.grid_size, self.grid_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.model = self._load_model()

        self._build_ui()
        self._predict_later()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot find checkpoint: {self.checkpoint_path}\n"
                "Run the MNIST training cells first to create proj_sim/outputs/mnist_cnn.pt."
            )
        model = LeNetMNIST(num_classes=10).to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model

    def _build_ui(self):
        outer = tk.Frame(self.root, padx=16, pady=16, bg="#111111")
        outer.pack(fill=tk.BOTH, expand=True)

        main = tk.Frame(outer, bg="#111111")
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, bg="#111111")
        left.pack(side=tk.LEFT, padx=(0, 18))

        self.pad = tk.Canvas(
            left,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=1,
            highlightbackground="#555555",
            cursor="crosshair",
        )
        self.pad.pack()
        self.pad.bind("<Button-1>", self._start_stroke)
        self.pad.bind("<B1-Motion>", self._draw_stroke)
        self.pad.bind("<ButtonRelease-1>", self._end_stroke)

        controls = tk.Frame(left, bg="#111111")
        controls.pack(fill=tk.X, pady=(12, 0))

        clear_btn = tk.Button(
            controls,
            text="Clear",
            command=self.clear,
            bg="#222222",
            fg="white",
            activebackground="#333333",
            activeforeground="white",
            relief=tk.FLAT,
            padx=14,
            pady=6,
        )
        clear_btn.pack(side=tk.LEFT)

        tk.Label(controls, text="Brush", bg="#111111", fg="white").pack(
            side=tk.LEFT, padx=(18, 8)
        )
        brush_slider = tk.Scale(
            controls,
            from_=1,
            to=5,
            orient=tk.HORIZONTAL,
            variable=self.brush_size,
            showvalue=True,
            length=150,
            bg="#111111",
            fg="white",
            troughcolor="#333333",
            highlightthickness=0,
        )
        brush_slider.pack(side=tk.LEFT)

        right = tk.Frame(main, bg="#111111")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.bar_canvas = tk.Canvas(
            right,
            width=380,
            height=280,
            bg="#111111",
            highlightthickness=0,
        )
        self.bar_canvas.pack(fill=tk.BOTH, expand=True)

        self.result_label = tk.Label(
            outer,
            text="Prediction: -",
            bg="#111111",
            fg="white",
            font=("Arial", 22, "bold"),
            pady=14,
        )
        self.result_label.pack(fill=tk.X)

    def _start_stroke(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self._paint_at(event.x, event.y)

    def _draw_stroke(self, event):
        if self.last_x is not None and self.last_y is not None:
            radius = self.brush_size.get() * self.canvas_scale / 2
            self.pad.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="white",
                width=max(1, int(radius * 2)),
                capstyle=tk.ROUND,
                smooth=True,
            )
            self._paint_line(self.last_x, self.last_y, event.x, event.y)
        self.last_x = event.x
        self.last_y = event.y
        self._predict_later()

    def _end_stroke(self, _event):
        self.last_x = None
        self.last_y = None
        self._predict_later(delay_ms=20)

    def _paint_at(self, x, y):
        gx = x / self.canvas_scale
        gy = y / self.canvas_scale
        r = self.brush_size.get() / 2
        self.draw.ellipse((gx - r, gy - r, gx + r, gy + r), fill=255)
        radius = self.brush_size.get() * self.canvas_scale / 2
        self.pad.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="white",
            outline="white",
        )
        self._predict_later()

    def _paint_line(self, x1, y1, x2, y2):
        width = max(1, self.brush_size.get())
        self.draw.line(
            (
                x1 / self.canvas_scale,
                y1 / self.canvas_scale,
                x2 / self.canvas_scale,
                y2 / self.canvas_scale,
            ),
            fill=255,
            width=width,
        )

    def _predict_later(self, delay_ms=45):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(delay_ms, self.predict)

    @torch.no_grad()
    def predict(self):
        self.after_id = None
        tensor = torch.tensor(list(self.image.getdata()), dtype=torch.float32)
        tensor = tensor.reshape(1, 1, self.grid_size, self.grid_size) / 255.0
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()
        pred = int(probs.argmax().item())
        self._draw_bars(probs.tolist(), pred)
        self.result_label.config(text=f"Prediction: {pred}    confidence: {probs[pred].item():.1%}")

    def _draw_bars(self, probs, pred):
        self.bar_canvas.delete("all")
        width = int(self.bar_canvas.winfo_width() or 380)
        height = int(self.bar_canvas.winfo_height() or 280)
        left_pad = 36
        bottom_pad = 34
        top_pad = 16
        gap = 7
        plot_w = width - left_pad - 12
        plot_h = height - top_pad - bottom_pad
        bar_w = max(14, (plot_w - gap * 9) / 10)

        for digit, prob in enumerate(probs):
            x0 = left_pad + digit * (bar_w + gap)
            x1 = x0 + bar_w
            bar_h = plot_h * prob
            y0 = top_pad + plot_h - bar_h
            y1 = top_pad + plot_h
            color = "#e53935" if digit == pred else "#9e9e9e"
            self.bar_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
            self.bar_canvas.create_text(
                (x0 + x1) / 2,
                height - 16,
                text=str(digit),
                fill="white",
                font=("Arial", 11),
            )
            self.bar_canvas.create_text(
                (x0 + x1) / 2,
                max(10, y0 - 9),
                text=f"{prob:.0%}",
                fill="white",
                font=("Arial", 9),
            )

        self.bar_canvas.create_line(left_pad, top_pad + plot_h, width - 8, top_pad + plot_h, fill="#555555")

    def clear(self):
        self.pad.delete("all")
        self.image = Image.new("L", (self.grid_size, self.grid_size), 0)
        self.draw = ImageDraw.Draw(self.image)
        self._predict_later(delay_ms=10)


def main():
    default_checkpoint = Path(__file__).resolve().parent / "outputs" / "mnist_cnn.pt"
    root = tk.Tk()
    try:
        MNISTDrawApp(root, default_checkpoint)
    except Exception as exc:
        messagebox.showerror("MNIST draw UI", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
