"""
digit_recognizer_compare.py
Sammenlign flere MNIST modeller - kompakt med kun Top 3
"""

import tkinter as tk
from tkinter import Frame, Label, Canvas, Button
from PIL import Image, ImageDraw, ImageOps
from keras.models import load_model
import numpy as np
import os


class DigitRecognizerCompare:
    def __init__(self, root, model_paths):
        self.root = root

        # Load alle modeller
        self.models = {}
        self.model_names = []

        print("Loading models...")
        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    self.models[name] = load_model(path)
                    self.model_names.append(name)
                    print(f"✅ Loaded: {name}")
                except Exception as e:
                    print(f"❌ Failed: {name} - {e}")

        if not self.models:
            raise ValueError("No models loaded!")

        # Setup window
        num_models = len(self.models)
        width = 400 + (num_models * 280)  # Canvas + models
        self.root.title("Håndskrevne Tal - Model Sammenligning")
        self.root.geometry(f"{width}x650")
        self.root.configure(bg='#2c3e50')

        self.after_id = None
        self.prediction_delay = 300

        # Track drawing state
        self.last_x = None
        self.last_y = None

        # PIL IMAGE
        self.canvas_width = 400
        self.canvas_height = 400
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Build UI
        self.build_ui()

    def build_ui(self):
        # Main container
        main_frame = Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # LEFT FRAME - Drawing canvas
        left_frame = Frame(main_frame, bg='#2c3e50')
        left_frame.pack(side=tk.LEFT, padx=(0, 20))

        # Title
        Label(
            left_frame,
            text="Tegn et tal (0-9)",
            font=('Helvetica', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        ).pack(pady=(0, 10))

        # CANVAS
        self.canvas = Canvas(
            left_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='white',
            cursor='cross',
            highlightthickness=2,
            highlightbackground='#34495e'
        )
        self.canvas.pack()

        # BIND MOUSE EVENTS
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        self.root.bind("<space>", lambda e: self.clear_canvas())

        # CLEAR BUTTON
        Button(
            left_frame,
            text="Ryd Canvas",
            command=self.clear_canvas,
            fg='black',
            bg='#e74c3c',
            font=('Helvetica', 14, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(pady=20)

        # RIGHT FRAME - Model predictions
        right_frame = Frame(main_frame, bg='#2c3e50')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create frame for each model
        self.model_widgets = {}

        for model_name in self.model_names:
            model_container = self.create_model_frame(right_frame, model_name)
            model_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

    def create_model_frame(self, parent, model_name):
        """Lav kompakt prediction frame med kun Top 3"""

        # Container
        container = Frame(parent, bg='#34495e', relief=tk.RAISED, borderwidth=2)

        # Model name
        Label(
            container,
            text=model_name.replace('_', ' ').title(),
            font=('Helvetica', 14, 'bold'),
            bg='#2c3e50',
            fg='white',
            pady=10
        ).pack(fill=tk.X)

        # TOP PREDICTION (big box)
        top_frame = Frame(container, bg='#2ecc71', height=120)
        top_frame.pack(fill=tk.X, padx=15, pady=15)
        top_frame.pack_propagate(False)

        Label(
            top_frame,
            text="Bedste Gæt:",
            font=('Helvetica', 11),
            bg='#2ecc71',
            fg='white'
        ).pack(pady=(12, 0))

        # Big digit
        top_digit_label = Label(
            top_frame,
            text="?",
            font=('Helvetica', 42, 'bold'),
            bg='#2ecc71',
            fg='white'
        )
        top_digit_label.pack()

        # Confidence
        confidence_label = Label(
            top_frame,
            text="---%",
            font=('Helvetica', 10),
            bg='#2ecc71',
            fg='white'
        )
        confidence_label.pack()

        # Divider
        Frame(container, bg='#2c3e50', height=2).pack(fill=tk.X, padx=15, pady=5)

        # TOP 3 PREDICTIONS
        top3_frame = Frame(container, bg='#34495e')
        top3_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        Label(
            top3_frame,
            text="Top 3 Predictions:",
            font=('Helvetica', 11, 'bold'),
            bg='#34495e',
            fg='#ecf0f1'
        ).pack(anchor='w', pady=(0, 8))

        top3_labels = []

        for i in range(3):
            pred_frame = Frame(top3_frame, bg='#2c3e50')
            pred_frame.pack(fill=tk.X, pady=4)

            # Medal + digit + percentage
            lbl = Label(
                pred_frame,
                text="(0.0%)",
                font=('Helvetica', 11),
                bg='#2c3e50',
                fg='white',
                anchor='w',
                padx=10,
                pady=5
            )
            lbl.pack(fill=tk.X)
            top3_labels.append(lbl)

        # Store references
        self.model_widgets[model_name] = {
            'top_frame': top_frame,
            'top_digit_label': top_digit_label,
            'confidence_label': confidence_label,
            'top3_labels': top3_labels
        }

        return container

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def paint(self, event):
        if self.last_x is None or self.last_y is None:
            self.last_x = event.x
            self.last_y = event.y
            return

        brush_size = 15

        # Draw on canvas
        self.canvas.create_line(
            self.last_x, self.last_y,
            event.x, event.y,
            width=brush_size * 2,
            fill='black',
            capstyle=tk.ROUND,
            smooth=True
        )

        # Draw on PIL image
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill='black',
            width=brush_size * 2,
            joint='curve'
        )

        # Schedule prediction
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(300, self.predict_all_models)

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None

    def preprocess_image(self):
        img = ImageOps.invert(self.image)
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        return img_array.reshape(1, 28, 28, 1)

    def predict_all_models(self):
        """Get predictions from all models"""
        img = self.preprocess_image()

        for model_name, model in self.models.items():
            prediction = model.predict(img, verbose=0)[0]

            widgets = self.model_widgets[model_name]

            # Top prediction
            top_digit = np.argmax(prediction)
            top_conf = prediction[top_digit] * 100

            widgets['top_digit_label'].config(text=str(top_digit))
            widgets['confidence_label'].config(text=f"{top_conf:.1f}%")

            # Color based on confidence
            if top_conf > 90:
                bg_color = '#2ecc71'  # Green
            elif top_conf > 70:
                bg_color = '#3498db'  # Blue
            elif top_conf > 50:
                bg_color = '#f39c12'  # Orange
            else:
                bg_color = '#e74c3c'  # Red

            widgets['top_frame'].config(bg=bg_color)
            widgets['top_digit_label'].config(bg=bg_color)
            widgets['confidence_label'].config(bg=bg_color)

            # Top 3 predictions
            top3_idx = np.argsort(prediction)[-3:][::-1]
            for i, idx in enumerate(top3_idx):
                prob = prediction[idx] * 100
                widgets['top3_labels'][i].config(
                    text=f"{idx} ({prob:.1f}%)"
                )

    def clear_canvas(self):
        """Clear canvas and reset predictions"""
        # Clear canvas
        self.canvas.delete("all")

        # Reset PIL image
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Reset drawing state
        self.last_x = None
        self.last_y = None

        # Reset all predictions
        for model_name, widgets in self.model_widgets.items():
            widgets['top_digit_label'].config(text="?")
            widgets['confidence_label'].config(text="---%")
            widgets['top_frame'].config(bg='#2ecc71')
            widgets['top_digit_label'].config(bg='#2ecc71')
            widgets['confidence_label'].config(bg='#2ecc71')


def main():
    # DEFINER DINE MODELLER HER
    model_paths = {
        'Baseline': './experiments/Aug 10/model_aug_none.keras',
        'Low Aug': './experiments/Aug 10/model_aug_low.keras',
        'Medium Aug': './experiments/Aug 10/model_aug_mid.keras',
        'High Aug': './experiments/Aug 10/model_aug_high.keras',
    }

    root = tk.Tk()
    app = DigitRecognizerCompare(root, model_paths)
    root.mainloop()


if __name__ == "__main__":
    main()

