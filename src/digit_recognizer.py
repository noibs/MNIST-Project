import tkinter as tk
from tkinter import Frame, Label, Canvas, Button
from PIL import Image, ImageDraw, ImageOps
from keras.models import load_model
import numpy as np


class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.model = load_model("./data/mnist_model.keras")
        self.root.title("Håndskrevne Tal - Neural Network Genkendelse")
        self.root.geometry("900x650")
        self.root.configure(bg='#2c3e50')
        self.after_id = None
        self.prediction_delay = 300

        # Track sidste drawing state
        self.last_x = None
        self.last_y = None

        # lav PIL image (til neural network processing)
        # Vi tegner på både tkinter Canvas og PIL Image
        self.canvas_width = 400
        self.canvas_height = 400
        self.image = Image.new(
            'L',  # Grayscale
            (self.canvas_width, self.canvas_height),
            'white'  # White background
        )
        self.draw = ImageDraw.Draw(self.image)

        # Main container
        main_frame = Frame(root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # LEFT FRAME
        left_frame = Frame(main_frame, bg='#2c3e50')
        left_frame.pack(side=tk.LEFT, padx=(0, 20))

        # Titel
        title_label = Label(
            left_frame,
            text="Tegn et tal (0-9)",
            font=('Helvetica', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=(0, 10))

        # CANVAS - hvor brugeren tegner
        self.canvas_width = 400
        self.canvas_height = 400

        self.canvas = Canvas(
            left_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='white',  # Hvid baggrund
            cursor='cross',  # Crosshair cursor
            highlightthickness=2,  # Border thickness
            highlightbackground='#34495e'  # Border color
        )
        self.canvas.pack()


        # Sæt event handlers
        # Venstre klik
        self.canvas.bind("<Button-1>", self.start_drawing)

        # Musse bevægelse med knap nede
        self.canvas.bind("<B1-Motion>", self.paint)

        # Slip venstre klik
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Mellemrum til at rydde canvas
        self.root.bind("<space>", lambda e: self.clear_canvas())

        # Ryd knap
        self.clear_button = Button(
            left_frame,
            text="Ryd Canvas",
            command=self.clear_canvas,
            fg='black',  # White text
            font=('Helvetica', 14, 'bold'),
            padx=20,
            pady=10,
            cursor='pointinghand'
        )
        self.clear_button.pack(pady=20)

        # RIGHT FRAME
        right_frame = Frame(main_frame, bg='#34495e', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)

        # Sandsynligheds titel
        pred_title = Label(
            right_frame,
            text="Real-time Predictions",
            font=('Helvetica', 20, 'bold'),
            bg='#34495e',
            fg='white'
        )
        pred_title.pack(pady=20)

        # Top sandsynligheds frame, viser det bedste gæt
        self.top_prediction_frame = Frame(
            right_frame,
            bg='#2ecc71',  # Grøn
            height=100
        )
        self.top_prediction_frame.pack(fill=tk.X, padx=20, pady=10)

        Label(
            self.top_prediction_frame,
            text="Bedste Gæt:",
            font=('Helvetica', 14),
            bg='#2ecc71',
            fg='white'
        ).pack(pady=(10, 0))

        # Det digit med højst sandsynlighed, stort tal med skriftstørrelse 48
        self.top_digit_label = Label(
            self.top_prediction_frame,
            text="?",  # Default value before prediction
            font=('Helvetica', 48, 'bold'),
            bg='#2ecc71',
            fg='white'
        )
        self.top_digit_label.pack()

        # Frame til sandsynlighedsbarer
        self.prob_frame = Frame(right_frame, bg='#34495e')
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Liste over sandsynlighedsbarer og labels
        self.prob_bars = []  # Canvas for drawing bars
        self.prob_percentages = []  # Labels showing percentages

        # Lav 10 sandsynlighedsbarer, en for hvert tal
        for i in range(10):
            # Frame til hvert tal
            digit_frame = Frame(self.prob_frame, bg='#34495e')
            digit_frame.pack(fill=tk.X, pady=5)

            # Tal label (viser "0:", "1:", osv.)
            digit_label = Label(
                digit_frame,
                text=f"{i}:",
                font=('Helvetica', 14, 'bold'),
                bg='#34495e',
                fg='white',
                width=2
            )
            digit_label.pack(side=tk.LEFT)

            # Container for baren (fast størrelse)
            bar_container = Frame(
                digit_frame,
                bg='#2c3e50',
                height=25,
                width=250
            )
            bar_container.pack(side=tk.LEFT, padx=10)
            bar_container.pack_propagate(False)

            # Canvas til at tegne baren
            bar = Canvas(
                bar_container,
                bg='#2c3e50',
                highlightthickness=0  # No border
            )
            bar.pack(fill=tk.BOTH, expand=True)
            self.prob_bars.append(bar)  # Gem reference

            # Procent label
            percentage = Label(
                digit_frame,
                text="0.0%",
                font=('Helvetica', 11),
                bg='#34495e',
                fg='white',
                width=6
            )
            percentage.pack(side=tk.LEFT)
            self.prob_percentages.append(percentage)  # Gem reference

        # Sæt alle sandsynligheder til 10% i starten
        self.update_probabilities([0.1] * 10)

    def start_drawing(self, event):
        """Bliver kaldt når vi starter med at tegne"""
        # Gemmer start position
        self.last_x = event.x
        self.last_y = event.y

    def paint(self, event):
        """Tegner på både tkinter Canvas og PIL Image"""
        if self.last_x is None or self.last_y is None:
            self.last_x = event.x
            self.last_y = event.y
            return

        brush_size = 15

        # Tegn på tkinter CANVAS (Det brugeren ser)
        self.canvas.create_line(
            self.last_x, self.last_y,
            event.x, event.y,
            width=brush_size * 2,
            fill='black',
            capstyle=tk.ROUND,
            smooth=True
        )

        # Tegn på PIL IMAGE (til det neural netværk)
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill='black',
            width=brush_size * 2,
            joint='curve'
        )

        # Stop tidligere planlagte prediction
        if self.after_id:
            self.root.after_cancel(self.after_id)

        # Lav en forsinkelse på 300ms før vi laver prediction
        self.after_id = self.root.after(300, self.predict_digit)

        # Opdaterer sidste position
        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        """Kaldet når brugeren slipper musse knappen"""
        self.last_x = None
        self.last_y = None

        #self.root.after(300, self.predict_digit)  # Predict efter 300ms


    def preprocess_image(self):
        """Forbereder PIL billedet til model prediction"""
        # Inverter farver (hvid baggrund, sort tal)
        img = ImageOps.invert(self.image)
        # Resize til 28x28 pixels
        img = img.resize((28, 28))
        # Konverter til numpy array og normaliser farver til [0, 1]
        img_array = np.array(img) / 255.0
        # Reshape til (1, 28, 28, 1) for model input
        return img_array.reshape(1, 28, 28, 1)

    def predict_digit(self):
        """Få model prediction og opdater GUI"""
        img = self.preprocess_image()
        # Få prediction fra model
        prediction = self.model.predict(img, verbose=0)[0]
        # Find tallet med højeste sandsynlighed
        digit = np.argmax(prediction)
        # Opdater top digit label
        self.top_digit_label.config(text=str(digit))
        self.update_probabilities(prediction)

    def update_probabilities(self, probabilities):
        """
        Opdaterer alle 10 sandsynlighedsbarer med nye værdier
        """
        # Find det tal med højeste sandsynlighed
        max_prob_idx = np.argmax(probabilities)

        # Opdater hver bar
        for i, (prob, bar, percentage_label) in enumerate(zip(
                probabilities,
                self.prob_bars,
                self.prob_percentages
        )):
            # Slet tidligere bar
            bar.delete("all")

            # Calculate bar width (max 250 pixels)
            # Beregn bredde af baren (maks 250 pixels)
            bar_width = int(prob * 250)

            # Vælg farve baseret på sandsynlighed og om det er den højeste
            if i == max_prob_idx:
                color = '#2ecc71'  # Grøn for højeste sandsynlighed
            elif prob > 0.5:
                color = '#3498db'  # Blå for høj sandsynlighed
            elif prob > 0.2:
                color = '#f39c12'  # Orange for medium sandsynlighed
            else:
                color = '#95a5a6'  # Grå for lav sandsynlighed

            # Tegn baren (rektangel)
            if bar_width > 0:
                bar.create_rectangle(
                    0, 0,  # Top-venstre hjørne (x, y)
                    bar_width, 25,  # Nederste-højre hjørne (x, y)
                    fill=color,  # Farve
                    outline=''  # Ingen outline
                )

            # Opdater procent label
            percentage_label.config(text=f"{prob * 100:.1f}%")

            # Fremhæv den højeste sandsynlighed
            if i == max_prob_idx:
                percentage_label.config(
                    fg='#2ecc71',  # Grøn text
                    font=('Helvetica', 12, 'bold')
                )
            else:
                percentage_label.config(
                    fg='white',
                    font=('Helvetica', 11)
                )

    def clear_canvas(self):
        """Ryd både Tkinter canvas og PIL image, og nulstil prediction"""
        # Ryd tkinter canvas
        self.canvas.delete("all")

        # Reset PIL image
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Reset tegne state
        self.last_x = None
        self.last_y = None

        # Reset predictions
        self.top_digit_label.config(text="?")
        self.top_prediction_frame.config(bg='#2ecc71')
        self.update_probabilities([0.1] * 10)


def main():
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
