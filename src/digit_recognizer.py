# digit_recognizer.py - FIXED VERSION FOR MAC
import tkinter as tk
from tkinter import Button, Canvas, Label, Frame
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import numpy as np
from keras.models import load_model
import os


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Håndskrevne Tal - Real-time Neural Network Genkendelse")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')

        self.root.bind('<space>', lambda e: self.clear_canvas())

        # Load den trænede model
        print("Indlæser model...")
        model_path = os.path.join('data', 'mnist_model.keras')
        self.model = load_model(model_path)
        print("Model indlæst!")

        # Flag til at tracke om vi skal opdatere predictions
        self.is_drawing = False
        self.prediction_delay = 300
        self.after_id = None

        # VIGTIGT: Gem sidste position for at tegne linjer
        self.last_x = None
        self.last_y = None

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

        # Bind musebevægelser - MED FIX!
        self.canvas.bind("<Button-1>", self.start_drawing)  # Når du klikker
        self.canvas.bind("<B1-Motion>", self.paint)  # Når du trækker
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)  # Når du slipper

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

        # Højre side - Predictions (samme som før)
        right_frame = Frame(main_frame, bg='#34495e', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)

        pred_title = Label(
            right_frame,
            text="Real-time Predictions",
            font=('Helvetica', 20, 'bold'),
            bg='#34495e',
            fg='white'
        )
        pred_title.pack(pady=20)

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

        self.prob_frame = Frame(right_frame, bg='#34495e')
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.prob_bars = []
        self.prob_labels = []
        self.prob_percentages = []

        for i in range(10):
            digit_frame = Frame(self.prob_frame, bg='#34495e')
            digit_frame.pack(fill=tk.X, pady=5)

            digit_label = Label(
                digit_frame,
                text=f"{i}:",
                font=('Helvetica', 14, 'bold'),
                bg='#34495e',
                fg='white',
                width=2
            )
            digit_label.pack(side=tk.LEFT)

            bar_container = Frame(digit_frame, bg='#2c3e50', height=25, width=250)
            bar_container.pack(side=tk.LEFT, padx=10)
            bar_container.pack_propagate(False)

            bar = Canvas(bar_container, bg='#2c3e50', highlightthickness=0)
            bar.pack(fill=tk.BOTH, expand=True)
            self.prob_bars.append(bar)

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

        self.update_probabilities([0.1] * 10)

    def start_drawing(self, event):
        """Når musen klikkes - gem startposition"""
        self.last_x = event.x
        self.last_y = event.y
        self.is_drawing = True

        # Tegn første punkt
        self.paint(event)

    def paint(self, event):
        """
        Tegn på canvas når musen bevæges
        FIX: Tegner LINJER mellem punkter i stedet for kun cirkler
        Dette løser "prikker" problemet på Mac!
        """
        if self.last_x is None or self.last_y is None:
            self.last_x = event.x
            self.last_y = event.y
            return

        # Tegn LINJE fra sidste position til nuværende position
        # Dette sikrer kontinuerlige streger selv ved hurtig bevægelse
        brush_size = 15

        # Tegn på tkinter canvas (linje i stedet for cirkel)
        self.canvas.create_line(
            self.last_x, self.last_y,
            event.x, event.y,
            width=brush_size * 2,
            fill='black',
            capstyle=tk.ROUND,  # Runde ender
            smooth=True  # Smooth linjer
        )

        # Tegn på PIL image (også linje)
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill='black',
            width=brush_size * 2,
            joint='curve'  # Smooth joints
        )

        # Opdater sidste position
        self.last_x = event.x
        self.last_y = event.y

        # Mark at vi er ved at tegne
        self.is_drawing = True

        # Schedule prediction update (debounced)
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(self.prediction_delay, self.predict_digit_auto)

    def stop_drawing(self, event):
        """Når musen slippes"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        self.predict_digit_auto()

    def clear_canvas(self):
        """Ryd canvas og PIL image"""
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Reset sidste position
        self.last_x = None
        self.last_y = None

        # Reset predictions til default
        self.top_digit_label.config(text="?")
        self.top_prediction_frame.config(bg='#2ecc71')
        self.update_probabilities([0.1] * 10)

    def preprocess_image(self):
        """Forbehandl billedet til MNIST format (28x28)"""
        img_array_check = np.array(self.image)
        if img_array_check.min() > 250:
            return None

        img = ImageOps.invert(self.image)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        img_array = np.array(img)
        img_array = img_array / 255.0

        # For CNN model: (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)

        return img_array

    def predict_digit_auto(self):
        """Lav automatisk prediction"""
        try:
            processed_img = self.preprocess_image()

            if processed_img is None:
                return

            prediction = self.model.predict(processed_img, verbose=0)[0]
            digit = np.argmax(prediction)
            confidence = prediction[digit]

            self.top_digit_label.config(text=str(digit))

            if confidence > 0.9:
                bg_color = '#2ecc71'
            elif confidence > 0.7:
                bg_color = '#f39c12'
            else:
                bg_color = '#e74c3c'

            self.top_prediction_frame.config(bg=bg_color)
            self.top_digit_label.config(bg=bg_color)

            self.update_probabilities(prediction)

        except Exception as e:
            print(f"Fejl i prediction: {e}")

    def update_probabilities(self, probabilities):
        """Opdater alle probability bars"""
        max_prob_idx = np.argmax(probabilities)

        for i, (prob, bar, percentage_label) in enumerate(zip(
                probabilities, self.prob_bars, self.prob_percentages
        )):
            bar.delete("all")
            bar_width = int(prob * 250)

            if i == max_prob_idx:
                color = '#2ecc71'
            elif prob > 0.5:
                color = '#3498db'
            elif prob > 0.2:
                color = '#f39c12'
            else:
                color = '#95a5a6'

            if bar_width > 0:
                bar.create_rectangle(
                    0, 0, bar_width, 25,
                    fill=color,
                    outline=''
                )

            percentage_label.config(text=f"{prob * 100:.1f}%")

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
