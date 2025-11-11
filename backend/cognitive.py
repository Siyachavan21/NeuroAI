import tensorflow as tf
import numpy as np
import os
import random
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import json
import time

# --- 1. CONFIGURATION ---
MODEL_PATH = 'E:/new laptop/mega project/integrationFinalProj/backend/Models/cognitive_model.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['focus', 'relax', 'stress'] 

# --- 2. IMPROVED TASK SUGGESTIONS ---
TASK_SUGGESTIONS = {
    'stress': [
        "Practice the 4-7-8 Breathing Technique: Inhale for 4s, hold for 7s, exhale for 8s.",
        "Use Mindful Observation: Find an object and spend 2 minutes noticing every detail about it.",
        "Try a 'Worry Dump': Spend 5 minutes writing down everything on your mind without filtering."
    ],
    'relax': [
        "Perform a Cat-Cow Yoga Stretch: Inhale as you drop your belly, exhale as you round your spine.",
        "Listen to Binaural Beats: Find a 'binaural beats for relaxation' track online and listen with headphones.",
        "Practice Visualization: Close your eyes and vividly imagine a peaceful place, like a beach or forest."
    ],
    'focus': [
        "Apply the 'Two-Minute Rule': If a task takes less than two minutes, do it immediately to build momentum.",
        "Create a 'Distraction List': When a distracting thought appears, write it down on paper to deal with later.",
        "Read one page of a book with the goal of summarizing it in one sentence afterward."
    ]
}

# --- 3. ENHANCED BUILT-IN GAME LIBRARY ---
# Each game now saves its final score to self.final_score and then closes itself.

class ZenBoxGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master); self.window.title("Zen Box")
        self.final_score = 0
        self.canvas = tk.Canvas(self.window, width=400, height=300, bg='#1F1F1F')
        self.canvas.pack(pady=10)
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#F0EAD6']
        self.box = self.canvas.create_rectangle(10, 125, 60, 175, fill=random.choice(self.colors), outline="")
        self.x_velocity = 1.5; self.time_left = 60
        self.time_label = tk.Label(self.window, text=f"Time for calm: {self.time_left}s", font=('Arial', 10))
        self.time_label.pack(pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click); self.move_box(); self.update_timer()
    def move_box(self):
        self.canvas.move(self.box, self.x_velocity, 0); pos = self.canvas.coords(self.box)
        if pos[2] >= 400 or pos[0] <= 0: self.x_velocity *= -1
        if self.time_left > 0: self.window.after(20, self.move_box)
    def on_canvas_click(self, event): self.canvas.itemconfig(self.box, fill=random.choice(self.colors))
    def update_timer(self):
        if self.time_left > 0:
            self.time_left -= 1; self.time_label.config(text=f"Time for calm: {self.time_left}s")
            self.window.after(1000, self.update_timer)
        else: self.final_score = 1; self.window.destroy()

class DrawingCanvasGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master); self.window.title("Drawing Canvas")
        self.final_score = 0; self.current_color = "black"
        controls_frame = tk.Frame(self.window); controls_frame.pack(pady=5)
        colors = ["black", "red", "green", "blue", "yellow", "orange", "purple"]
        for color in colors:
            tk.Button(controls_frame, bg=color, width=2, command=lambda c=color: self.set_color(c)).pack(side="left", padx=2)
        self.canvas = tk.Canvas(self.window, width=500, height=400, bg='white'); self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        tk.Button(self.window, text="Clear", command=lambda: self.canvas.delete("all")).pack(side="left", padx=10, pady=5)
        tk.Button(self.window, text="Finish", command=self.end_game).pack(side="right", padx=10, pady=5)
    def set_color(self, color): self.current_color = color
    def draw(self, event): self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill=self.current_color, outline=self.current_color)
    def end_game(self): self.final_score = 1; self.window.destroy()

class TargetPracticeGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master); self.window.title("Target Practice")
        self.final_score = 0
        self.canvas = tk.Canvas(self.window, width=500, height=400, bg='black'); self.canvas.pack()
        self.score, self.time_left, self.combo = 0, 30, 0
        self.score_label = tk.Label(self.window, text=f"Score: {self.score}", font=('Arial', 14)); self.score_label.pack(side="left", padx=10)
        self.combo_label = tk.Label(self.window, text=f"Combo: x{self.combo}", font=('Arial', 14)); self.combo_label.pack(side="left", padx=10)
        self.time_label = tk.Label(self.window, text=f"Time: {self.time_left}", font=('Arial', 14)); self.time_label.pack(side="right", padx=10)
        self.canvas.bind("<Button-1>", self.on_miss); self.spawn_target(); self.update_timer()
    def spawn_target(self):
        self.canvas.delete("all"); x, y = random.randint(30, 470), random.randint(30, 370)
        self.target_radius = 30
        self.target = self.canvas.create_oval(x-self.target_radius, y-self.target_radius, x+self.target_radius, y+self.target_radius, fill='red', tags="target")
        self.canvas.tag_bind("target", "<Button-1>", self.on_target_click); self.shrink_target()
    def shrink_target(self):
        if self.time_left > 0 and self.target_radius > 5:
            self.target_radius -= 0.5; coords = self.canvas.coords(self.target)
            cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
            self.canvas.coords(self.target, cx-self.target_radius, cy-self.target_radius, cx+self.target_radius, cy+self.target_radius)
            self.window.after(50, self.shrink_target)
        elif self.time_left > 0: self.on_miss(None)
    def on_target_click(self, event):
        self.combo += 1; points = int(35 - self.target_radius) * self.combo
        self.score += points
        self.score_label.config(text=f"Score: {self.score}"); self.combo_label.config(text=f"Combo: x{self.combo}")
        self.spawn_target()
    def on_miss(self, event):
        self.combo = 0; self.score -= 5
        self.score_label.config(text=f"Score: {self.score}"); self.combo_label.config(text=f"Combo: x{self.combo}")
        if self.time_left > 0: self.spawn_target()
    def update_timer(self):
        if self.time_left > 0:
            self.time_left -= 1; self.time_label.config(text=f"Time: {self.time_left}")
            self.window.after(1000, self.update_timer)
        else: self.end_game()
    def end_game(self): self.final_score = self.score; self.window.destroy()

class TypingSpeedGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master); self.window.title("Typing Speed Test")
        self.final_score = 0
        self.words = ["python", "focus", "challenge", "cognitive", "model", "develop", "accuracy", "breathe", "relax"]
        self.time_left, self.score, self.total_chars = 45, 0, 0
        tk.Label(self.window, text=f"Type as many words as you can in {self.time_left} seconds!", font=('Arial', 14)).pack(pady=10)
        self.time_label = tk.Label(self.window, text=f"Time: {self.time_left}", font=('Arial', 14)); self.time_label.pack()
        self.score_label = tk.Label(self.window, text="Score: 0", font=('Arial', 14)); self.score_label.pack()
        self.word_label = tk.Label(self.window, text="", font=('Arial', 24, 'bold'), fg='blue'); self.word_label.pack(pady=10)
        self.entry = tk.Entry(self.window, font=('Arial', 18), justify='center'); self.entry.pack(pady=10, padx=10)
        self.entry.bind("<Return>", self.check_word); self.next_word(); self.update_timer()
    def next_word(self):
        self.entry.delete(0, tk.END); self.current_word = random.choice(self.words)
        self.word_label.config(text=self.current_word)
    def check_word(self, event):
        if self.time_left > 0 and self.entry.get() == self.current_word:
            self.score += 1; self.total_chars += len(self.current_word)
            self.score_label.config(text=f"Score: {self.score}"); self.next_word()
    def update_timer(self):
        if self.time_left > 0:
            self.time_left -= 1; self.time_label.config(text=f"Time: {self.time_left}")
            self.window.after(1000, self.update_timer)
        else: self.end_game()
    def end_game(self):
        self.entry.config(state='disabled')
        wpm = (self.total_chars / 5) / (45 / 60) if self.total_chars > 0 else 0
        self.final_score = wpm; self.window.destroy()


GAME_LIBRARY = {
    'stress': [ZenBoxGame, DrawingCanvasGame], 
    'relax': [DrawingCanvasGame, ZenBoxGame], #
    'focus': [TargetPracticeGame, TypingSpeedGame]
}

# Global variable to track active game window
active_game_window = None


# --- 4. NEW: The Post-Game Analysis Window ---
class GameAnalysisWindow:
    def __init__(self, master, state, game_class, score):
        self.window = tk.Toplevel(master)
        self.window.title("Game Report")
        self.window.config(padx=30, pady=20, bg="#F0F0F0")

        tk.Label(self.window, text="Game Performance Report", font=('Arial', 18, 'bold'), bg="#F0F0F0").pack(pady=(0, 20))

        score_text, insight, takeaway = self.get_analysis_message(state, game_class, score)
        
        # Performance Section
        tk.Label(self.window, text="Your Performance:", font=('Arial', 12, 'bold'), bg="#F0F0F0").pack(anchor="w")
        tk.Label(self.window, text=score_text, font=('Arial', 12), bg="#F0F0F0").pack(anchor="w", pady=(0, 10))
        
        # Insight Section
        tk.Label(self.window, text="Our Insight:", font=('Arial', 12, 'bold'), bg="#F0F0F0").pack(anchor="w")
        tk.Label(self.window, text=insight, font=('Arial', 12, 'italic'), wraplength=400, justify="left", bg="#F0F0F0").pack(anchor="w", pady=(0, 10))

        # Takeaway Section
        tk.Label(self.window, text="Your Takeaway:", font=('Arial', 12, 'bold'), bg="#F0F0F0").pack(anchor="w")
        tk.Label(self.window, text=takeaway, font=('Arial', 12), wraplength=400, justify="left", bg="#F0F0F0").pack(anchor="w")

        tk.Button(self.window, text="Finish Session", command=master.destroy, font=('Arial', 12, 'bold'), bg="#4CAF50", fg="white", relief="flat").pack(pady=20)
        
    def get_analysis_message(self, state, game_class, score):
        if game_class in [ZenBoxGame, DrawingCanvasGame]:
            score_text = "Session Completed"
            insight = f"You took a moment for a calming activity, which is a powerful way to respond to feelings of {state}."
            takeaway = "Remember this feeling of calm. If you feel overwhelmed later, try to recall this moment or take 60 seconds to focus on your breath."
            return score_text, insight, takeaway
        
        elif game_class == TargetPracticeGame:
            score_text = f"Final Score: {score}"
            if score > 1000:
                insight = "Your incredible score shows exceptional reflexes and precision. You are clearly in a state of high focus right now."
                takeaway = "Now is a perfect time to tackle a challenging task that requires deep work. Your mind is warmed up and ready."
            elif score > 500:
                insight = "A great score! You have a sharp eye and quick reaction time, indicating a strong ability to concentrate."
                takeaway = "Try to bring this same level of engagement to your next task for just 10-15 minutes."
            else:
                insight = "A solid effort! Games like this are a fantastic workout for your focus 'muscles'."
                takeaway = "The key to improving focus is consistent practice. You're on the right track!"
            return score_text, insight, takeaway

        elif game_class == TypingSpeedGame:
            score_text = f"Final Speed: {score:.2f} WPM"
            if score > 60:
                insight = "An outstanding WPM score! This level of speed and accuracy is a clear sign of being 'in the zone'."
                takeaway = "Your mind is sharp and ready for complex tasks. Use this momentum to your advantage."
            elif score > 40:
                insight = "Excellent work! A speed above 40 WPM is highly productive and shows a strong mind-body connection."
                takeaway = "Challenge yourself with a short writing or coding task to make the most of your current focus."
            else:
                insight = "A very good result! You are building the foundations of quick and accurate focus."
                takeaway = "To boost your focus further, try minimizing distractions in your environment for the next 30 minutes."
            return score_text, insight, takeaway
        
        return "", "", ""


# --- 5. The GUI Results Window ---
class ResultsWindow:
    def __init__(self, root, image_path, state, confidence, task):
        self.root = root; self.root.title("Cognitive Analysis Results")
        self.state = state
        BG_COLOR, TEXT_COLOR = "#2E2E2E", "#EAEAEA"
        FONT_NORMAL, FONT_BOLD = ("Arial", 12), ("Arial", 16, "bold")
        STATE_COLORS = {'stress': "#FF6B6B", 'relax': "#4ECDC4", 'focus': "#45B7D1"}
        self.root.config(bg=BG_COLOR, padx=20, pady=20)
        image_frame = tk.Frame(self.root, bg=BG_COLOR)
        image_frame.pack(side="left", padx=(0, 20))
        img = Image.open(image_path)
        img.thumbnail((250, 250))
        self.photo_img = ImageTk.PhotoImage(img)
        image_label = tk.Label(image_frame, image=self.photo_img, bg=BG_COLOR)
        image_label.image = self.photo_img; image_label.pack()
        results_frame = tk.Frame(self.root, bg=BG_COLOR)
        results_frame.pack(side="left", fill="both", expand=True)
        tk.Label(results_frame, text="Analysis Complete", font=("Arial", 20, "bold"), fg=TEXT_COLOR, bg=BG_COLOR).pack(anchor="w")
        tk.Label(results_frame, text=f"DETECTED STATE:", font=FONT_NORMAL, fg=TEXT_COLOR, bg=BG_COLOR).pack(anchor="w", pady=(20, 0))
        tk.Label(results_frame, text=f"{state.upper()}", font=("Arial", 28, "bold"), fg=STATE_COLORS.get(state), bg=BG_COLOR).pack(anchor="w")
        tk.Label(results_frame, text=f"Confidence: {confidence:.2f}%", font=FONT_NORMAL, fg=TEXT_COLOR, bg=BG_COLOR).pack(anchor="w")
        tk.Label(results_frame, text="RECOMMENDED TASK:", font=FONT_NORMAL, fg=TEXT_COLOR, bg=BG_COLOR).pack(anchor="w", pady=(20, 0))
        tk.Label(results_frame, text=task, font=FONT_NORMAL, fg=TEXT_COLOR, bg=BG_COLOR, wraplength=300, justify="left").pack(anchor="w")
        tk.Button(results_frame, text=f"Play a Game for {state.capitalize()}", command=self.launch_game, font=FONT_BOLD, bg="#555555", fg=TEXT_COLOR, relief="flat", padx=10, pady=5).pack(pady=(30, 0), anchor="w")

    def launch_game(self):
        self.root.withdraw()
        available_games = GAME_LIBRARY[self.state]
        chosen_game_class = random.choice(available_games)
        
        # Close any existing game window before opening new one
        global active_game_window
        if active_game_window and active_game_window.window.winfo_exists():
            active_game_window.window.destroy()
        
        game_instance = chosen_game_class(self.root)
        active_game_window = game_instance
        self.root.wait_window(game_instance.window)
        GameAnalysisWindow(self.root, self.state, chosen_game_class, game_instance.final_score)

# --- 6. CORE ML FUNCTION ---
def get_recommendations(image_path, model):
    if not os.path.exists(image_path): return "Error: Image not found", 0, None
    img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    tf.get_logger().setLevel('ERROR')
    predictions = model.predict(img_array)
    tf.get_logger().setLevel('INFO')
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    task = random.choice(TASK_SUGGESTIONS[predicted_class])
    return predicted_class, confidence, task

# --- 7. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
            accuracy = metadata.get('validation_accuracy')
            if accuracy: print(f"🧠 Model Performance: This model has a validation accuracy of {accuracy:.2%}.")
    except FileNotFoundError:
        print("⚠️ Model metadata not found. Please run the train_model.py script to generate it.")
    
    print("\n----------------------------------------------------")
    
    root = tk.Tk()
    root.withdraw()
    image_to_test = filedialog.askopenfilename(title="Select an Image for Cognitive Analysis", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])

    if not image_to_test:
        print("No image selected. Exiting.")
    elif not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
    else:
        print("Loading model and analyzing image...")
        cognitive_model = tf.keras.models.load_model(MODEL_PATH)
        state, confidence, task = get_recommendations(image_to_test, cognitive_model)
        
        if state and "Error" not in state:
            results_root = tk.Toplevel(root)
            app = ResultsWindow(results_root, image_to_test, state, confidence, task)
            root.mainloop()
        else:
            print(f"Error: Could not get a recommendation.")