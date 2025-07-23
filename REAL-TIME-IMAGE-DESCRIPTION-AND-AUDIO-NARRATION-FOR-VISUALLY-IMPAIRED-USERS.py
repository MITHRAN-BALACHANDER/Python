from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pyttsx3
import random
try:
    model = YOLO("yolov8x.pt")
    print("✅ YOLOv8x model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    model = None

engine = pyttsx3.init()

engine.setProperty('rate', 150)  # Slower speech

# Set volume (0.0 to 1.0)
engine.setProperty('volume', 1.0)  

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  #male 0 female 1

def detect_objects(image_path):
    if not model:
        return []
    results = model(image_path)[0]
    labels = [model.names[int(cls)] for cls in results.boxes.cls]
    return list(set(labels))

def generate_prompt_from_objects(objects):
    if not objects:
        return "No objects detected in the image."

    adjectives = ['beautiful', 'fascinating', 'adorable', 'majestic', 'vibrant', 'peaceful']
    actions = ['sitting', 'running', 'jumping', 'walking', 'flying', 'playing', 'standing']

    adj = random.choice(adjectives)
    act = random.choice(actions)

    if len(objects) == 1:
        return f"A {adj} {objects[0]} is {act} in this photo."
    elif len(objects) == 2:
        return f"Shows a {adj} scene with a {objects[0]} and a {objects[1]}."
    else:
        middle = ', '.join(objects[:-1])
        last = objects[-1]
        return f"A {adj} scene featuring {middle}, and a {last} {act}."

# GUI App
class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Features Generator")
        self.root.geometry("800x800")

        self.image_path = None
        self.caption_prompt = ""  # Only caption, no detected objects text

        self.label = tk.Label(root, text="Select an image to begin.", font=("Arial", 14))
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=500, height=400)
        self.canvas.pack()

        self.select_btn = tk.Button(root, text="📂 Select Image", command=self.select_image)
        self.select_btn.pack(pady=10)

        self.detect_btn = tk.Button(root, text="🔍 Generate", command=self.run_detection)
        self.detect_btn.pack(pady=5)

        self.listen_btn = tk.Button(root, text="🔊 Listen", command=self.listen_prompt)
        self.listen_btn.pack(pady=5)

        self.prompt_text = tk.Text(root, height=5, wrap='word')
        self.prompt_text.pack(pady=10)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if self.image_path:
            self.display_image(self.image_path)
            self.prompt_text.delete('1.0', tk.END)
            self.label.config(text="Image loaded. Click 'Generate'.")

    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((500, 400))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(250, 200, image=self.tk_img)

    def run_detection(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        detected = detect_objects(self.image_path)
        if not detected:
            messagebox.showinfo("Result", "No objects detected.")
            self.prompt_text.delete('1.0', tk.END)
            self.prompt_text.insert(tk.END, "No objects detected.")
            self.caption_prompt = "No objects detected."
            return

        caption = generate_prompt_from_objects(detected)
        self.caption_prompt = caption  

        result = f"{caption}\n\nDetected objects: {', '.join(detected)}"
        self.prompt_text.delete('1.0', tk.END)
        self.prompt_text.insert(tk.END, result)
        self.label.config(text="Generation complete Click Listen.")

    def listen_prompt(self):
        if not self.caption_prompt:
            messagebox.showwarning("Warning", "Nothing to read. Generate a caption first!")
            return
        engine.say(self.caption_prompt)
        engine.runAndWait()

# Run GUI
if __name__ == '__main__':
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()
