# digit_recognizer.py - REAL-TIME VERSION
import tkinter as tk
from tkinter import Button, Canvas, Label, Frame
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
from keras.models import load_model
import os


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Håndskrevne Tal")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')

        self.root.bind('<space>', lambda e: self.clear_canvas())
        self.root.bind('<Escape>', lambda e: self.root.quit())

        # Load den trænede model
        print("Indlæser model...")
        model_path = os.path.join('data', 'mnist_model.keras')
        self.model = load_model(model_path)
        print("Model indlæst!")

        # Flag til at tracke om vi skal opdatere predictions
        self.is_drawing = False
        self.prediction_delay = 200  # millisekunder mellem predictions
        self.after_id = None

        # Main container
        main_frame = Frame(root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Venstre side - Canvas
        left_frame = Frame(main_frame, bg='#2c3e50')
        left_frame.pack(side=tk.LEFT, padx=(0, 20))

        # Title
        title_label = Label(
            left_frame,
            text="Tegn et tal (0-9)",
            font=('Helvetica', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=(0, 10))

        # Canvas til at tegne på
        self.canvas_width = 400
        self.canvas_height = 400
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

        # PIL image til at gemme tegningen
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Bind musebevægelser til tegning og real-time prediction
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Clear knap
        self.clear_button = Button(
            left_frame,
            text="Ryd Canvas",
            command=self.clear_canvas,
            bg='#e74c3c',
            fg='white',
            font=('Helvetica', 14, 'bold'),
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.clear_button.pack(pady=20)

        # Højre side - Predictions
        right_frame = Frame(main_frame, bg='#34495e', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)

        # Prediction title
        pred_title = Label(
            right_frame,
            text="Real-time Predictions",
            font=('Helvetica', 20, 'bold'),
            bg='#34495e',
            fg='white'
        )
        pred_title.pack(pady=20)

        # Top prediction display
        self.top_prediction_frame = Frame(right_frame, bg='#2ecc71', height=100)
        self.top_prediction_frame.pack(fill=tk.X, padx=20, pady=10)

        Label(
            self.top_prediction_frame,
            text="Bedste Gæt:",
            font=('Helvetica', 14),
            bg='#2ecc71',
            fg='white'
        ).pack(pady=(10, 0))

        self.top_digit_label = Label(
            self.top_prediction_frame,
            text="?",
            font=('Helvetica', 48, 'bold'),
            bg='#2ecc71',
            fg='white'
        )
        self.top_digit_label.pack()

        # Probability bars frame
        self.prob_frame = Frame(right_frame, bg='#34495e')
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create probability bars for each digit (0-9)
        self.prob_bars = []
        self.prob_labels = []
        self.prob_percentages = []

        for i in range(10):
            digit_frame = Frame(self.prob_frame, bg='#34495e')
            digit_frame.pack(fill=tk.X, pady=5)

            # Digit label
            digit_label = Label(
                digit_frame,
                text=f"{i}:",
                font=('Helvetica', 14, 'bold'),
                bg='#34495e',
                fg='white',
                width=2
            )
            digit_label.pack(side=tk.LEFT)

            # Progress bar container
            bar_container = Frame(digit_frame, bg='#2c3e50', height=25, width=250)
            bar_container.pack(side=tk.LEFT, padx=10)
            bar_container.pack_propagate(False)

            # Progress bar
            bar = Canvas(bar_container, bg='#2c3e50', highlightthickness=0)
            bar.pack(fill=tk.BOTH, expand=True)
            self.prob_bars.append(bar)

            # Percentage label
            percentage = Label(
                digit_frame,
                text="0.0%",
                font=('Helvetica', 11),
                bg='#34495e',
                fg='white',
                width=6
            )
            percentage.pack(side=tk.LEFT)
            self.prob_percentages.append(percentage)

        # Initial state
        self.update_probabilities([0.1] * 10)  # Equal probabilities initially

    def paint(self, event):
        """Tegn på canvas når musen bevæges"""
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)

        # Tegn på tkinter canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)

        # Tegn på PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

        # Mark at vi er ved at tegne
        self.is_drawing = True

        # Schedule prediction update (debounced)
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(self.prediction_delay, self.predict_digit_auto)

    def stop_drawing(self, event):
        """Når musen slippes, lav en final prediction"""
        self.is_drawing = False
        self.predict_digit_auto()

    def clear_canvas(self):
        """Ryd canvas og PIL image"""
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Reset predictions til default
        self.top_digit_label.config(text="?")
        self.top_prediction_frame.config(bg='#2ecc71')
        self.update_probabilities([0.1] * 10)

    def preprocess_image(self):
        """Forbehandl billedet til MNIST format (28x28)"""
        # Check if canvas is empty
        img_array_check = np.array(self.image)
        if img_array_check.min() > 250:
            return None

        # Inverter farver
        img = ImageOps.invert(self.image)

        # Resize til 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Anvend Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Konverter til numpy array og normaliser
        img_array = np.array(img)
        img_array = img_array / 255.0

        # VIGTIGT: CNN modeller forventer (batch, height, width, channels)
        img_array = img_array.reshape(1, 28, 28, 1)  # Ikke (1, 784)!

        return img_array

    def predict_digit_auto(self):
        """Lav automatisk prediction (kaldt under tegning)"""
        try:
            # Forbehandl billedet
            processed_img = self.preprocess_image()

            if processed_img is None:
                # Canvas er tom
                return

            # Lav forudsigelse
            prediction = self.model.predict(processed_img, verbose=0)[0]

            # Find det mest sandsynlige tal
            digit = np.argmax(prediction)
            confidence = prediction[digit]

            # Opdater top prediction
            self.top_digit_label.config(text=str(digit))

            # Color code based on confidence
            if confidence > 0.9:
                bg_color = '#2ecc71'  # Green
            elif confidence > 0.7:
                bg_color = '#f39c12'  # Orange
            else:
                bg_color = '#e74c3c'  # Red

            self.top_prediction_frame.config(bg=bg_color)
            self.top_digit_label.config(bg=bg_color)

            # Opdater probability bars
            self.update_probabilities(prediction)

        except Exception as e:
            print(f"Fejl i prediction: {e}")

    def update_probabilities(self, probabilities):
        """Opdater alle probability bars"""
        max_prob_idx = np.argmax(probabilities)

        for i, (prob, bar, percentage_label) in enumerate(zip(
                probabilities, self.prob_bars, self.prob_percentages
        )):
            # Clear previous bar
            bar.delete("all")

            # Calculate bar width (max 250 pixels)
            bar_width = int(prob * 250)

            # Color gradient based on probability
            if i == max_prob_idx:
                color = '#2ecc71'  # Green for highest
            elif prob > 0.5:
                color = '#3498db'  # Blue for high
            elif prob > 0.2:
                color = '#f39c12'  # Orange for medium
            else:
                color = '#95a5a6'  # Gray for low

            # Draw bar
            if bar_width > 0:
                bar.create_rectangle(
                    0, 0, bar_width, 25,
                    fill=color,
                    outline=''
                )

            # Update percentage label
            percentage_label.config(text=f"{prob * 100:.1f}%")

            # Highlight the highest probability
            if i == max_prob_idx:
                percentage_label.config(fg='#2ecc71', font=('Helvetica', 12, 'bold'))
            else:
                percentage_label.config(fg='white', font=('Helvetica', 11))


def main():
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
