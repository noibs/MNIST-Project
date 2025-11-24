import tkinter as tk
from tkinter import Frame, Label, Canvas, Button
from PIL import Image, ImageDraw, ImageOps, ImageFilter
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
        self.prediction_delay = 300  # milliseconds

        # Track drawing state
        self.last_x = None
        self.last_y = None

        # CREATE PIL IMAGE (for neural network processing)
        # We draw on BOTH tkinter Canvas AND PIL Image
        self.canvas_width = 400
        self.canvas_height = 400
        self.image = Image.new(
            'L',  # Grayscale mode
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

        # Title
        title_label = Label(
            left_frame,
            text="Tegn et tal (0-9)",
            font=('Helvetica', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=(0, 10))

        # CANVAS - where users draw
        self.canvas_width = 400
        self.canvas_height = 400

        self.canvas = Canvas(
            left_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg='white',  # White background for drawing
            cursor='cross',  # Crosshair cursor
            highlightthickness=2,  # Border width
            highlightbackground='#34495e'  # Border color
        )
        self.canvas.pack()


        # BIND MOUSE EVENTS to canvas
        # When user clicks (presses mouse button)
        self.canvas.bind("<Button-1>", self.start_drawing)

        # When user drags (moves mouse with button pressed)
        self.canvas.bind("<B1-Motion>", self.paint)

        # When user releases mouse button
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.root.bind("<space>", lambda e: self.clear_canvas())

        # CLEAR BUTTON
        self.clear_button = Button(
            left_frame,
            text="Ryd Canvas",
            command=self.clear_canvas,  # Function to call when clicked
            bg='#e74c3c',  # Red background
            fg='white',  # White text
            font=('Helvetica', 14, 'bold'),
            padx=20,
            pady=10,
            cursor='pointinghand'  # Hand cursor on hover
        )
        self.clear_button.pack(pady=20)

        # RIGHT FRAME
        right_frame = Frame(main_frame, bg='#34495e', width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        right_frame.pack_propagate(False)

        # Predictions title
        pred_title = Label(
            right_frame,
            text="Real-time Predictions",
            font=('Helvetica', 20, 'bold'),
            bg='#34495e',
            fg='white'
        )
        pred_title.pack(pady=20)

        # TOP PREDICTION BOX - shows the best guess
        self.top_prediction_frame = Frame(
            right_frame,
            bg='#2ecc71',  # Green background
            height=100
        )
        self.top_prediction_frame.pack(fill=tk.X, padx=20, pady=10)

        # Label saying "Bedste Gæt:"
        Label(
            self.top_prediction_frame,
            text="Bedste Gæt:",
            font=('Helvetica', 14),
            bg='#2ecc71',
            fg='white'
        ).pack(pady=(10, 0))

        # The actual digit prediction (big number)
        self.top_digit_label = Label(
            self.top_prediction_frame,
            text="?",  # Default value before prediction
            font=('Helvetica', 48, 'bold'),
            bg='#2ecc71',
            fg='white'
        )
        self.top_digit_label.pack()

        # PROBABILITY BARS CONTAINER
        self.prob_frame = Frame(right_frame, bg='#34495e')
        self.prob_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Lists to store references to widgets
        self.prob_bars = []  # Canvas for drawing bars
        self.prob_percentages = []  # Labels showing percentages

        # Create 10 probability bars (one for each digit 0-9)
        for i in range(10):
            # Container for this digit's row
            digit_frame = Frame(self.prob_frame, bg='#34495e')
            digit_frame.pack(fill=tk.X, pady=5)

            # Digit label (shows "0:", "1:", etc.)
            digit_label = Label(
                digit_frame,
                text=f"{i}:",
                font=('Helvetica', 14, 'bold'),
                bg='#34495e',
                fg='white',
                width=2
            )
            digit_label.pack(side=tk.LEFT)

            # Bar container (fixed size)
            bar_container = Frame(
                digit_frame,
                bg='#2c3e50',
                height=25,
                width=250
            )
            bar_container.pack(side=tk.LEFT, padx=10)
            bar_container.pack_propagate(False)

            # Canvas for drawing the progress bar
            bar = Canvas(
                bar_container,
                bg='#2c3e50',
                highlightthickness=0  # No border
            )
            bar.pack(fill=tk.BOTH, expand=True)
            self.prob_bars.append(bar)  # Save reference

            # Percentage label (shows "0.0%", etc.)
            percentage = Label(
                digit_frame,
                text="0.0%",
                font=('Helvetica', 11),
                bg='#34495e',
                fg='white',
                width=6
            )
            percentage.pack(side=tk.LEFT)
            self.prob_percentages.append(percentage)  # Save reference

        # Initialize with default probabilities
        self.update_probabilities([0.1] * 10)  # Equal 10% for all

    def start_drawing(self, event):
        """Called when user first clicks on canvas"""
        # Save the starting position
        self.last_x = event.x
        self.last_y = event.y

    def paint(self, event):
        """Draw on both Canvas AND PIL Image"""
        if self.last_x is None or self.last_y is None:
            self.last_x = event.x
            self.last_y = event.y
            return

        brush_size = 15

        # Draw on TKINTER CANVAS (what user sees)
        self.canvas.create_line(
            self.last_x, self.last_y,
            event.x, event.y,
            width=brush_size * 2,
            fill='black',
            capstyle=tk.ROUND,
            smooth=True
        )

        # Draw on PIL IMAGE (for neural network)
        self.draw.line(
            [self.last_x, self.last_y, event.x, event.y],
            fill='black',
            width=brush_size * 2,
            joint='curve'
        )

        # Cancel previous scheduled prediction
        if self.after_id:
            self.root.after_cancel(self.after_id)

        # Schedule NEW prediction in 300ms
        self.after_id = self.root.after(300, self.predict_digit)

        # Update position
        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        """Called when user releases mouse button"""
        self.last_x = None
        self.last_y = None

        #self.root.after(300, self.predict_digit)  # Debounced prediction


    def preprocess_image(self):
        img = ImageOps.invert(self.image)
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        return img_array.reshape(1, 28, 28, 1)

    def predict_digit(self):
        img = self.preprocess_image()
        prediction = self.model.predict(img, verbose=0)[0]
        digit = np.argmax(prediction)
        self.top_digit_label.config(text=str(digit))
        self.update_probabilities(prediction)

    def update_probabilities(self, probabilities):
        """Update all 10 probability bars with new values

        Args:
            probabilities: List of 10 floats (0.0 to 1.0)
        """
        # Find which digit has highest probability
        import numpy as np
        max_prob_idx = np.argmax(probabilities)

        # Update each bar
        for i, (prob, bar, percentage_label) in enumerate(zip(
                probabilities,
                self.prob_bars,
                self.prob_percentages
        )):
            # Clear previous bar
            bar.delete("all")

            # Calculate bar width (max 250 pixels)
            bar_width = int(prob * 250)

            # Choose color based on probability and if it's the max
            if i == max_prob_idx:
                color = '#2ecc71'  # Green for highest probability
            elif prob > 0.5:
                color = '#3498db'  # Blue for high probability
            elif prob > 0.2:
                color = '#f39c12'  # Orange for medium
            else:
                color = '#95a5a6'  # Gray for low

            # Draw the bar (rectangle)
            if bar_width > 0:
                bar.create_rectangle(
                    0, 0,  # Top-left corner (x, y)
                    bar_width, 25,  # Bottom-right corner (x, y)
                    fill=color,  # Fill color
                    outline=''  # No outline
                )

            # Update percentage text
            percentage_label.config(text=f"{prob * 100:.1f}%")

            # Highlight the highest probability
            if i == max_prob_idx:
                percentage_label.config(
                    fg='#2ecc71',  # Green text
                    font=('Helvetica', 12, 'bold')
                )
            else:
                percentage_label.config(
                    fg='white',
                    font=('Helvetica', 11)
                )

    def clear_canvas(self):
        """Clear both Canvas and PIL Image"""
        # Clear tkinter canvas
        self.canvas.delete("all")

        # Reset PIL image
        self.image = Image.new('L', (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Reset drawing state
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
