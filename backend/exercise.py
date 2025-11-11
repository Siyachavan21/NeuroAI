from flask import Flask, request, jsonify
from flask_cors import CORS
import tkinter as tk
import multiprocessing
import random

app = Flask(__name__)
CORS(app)

# Track if a game is currently running
game_running = multiprocessing.Value('i', 0)


# --- GAME CLASSES ---

class ZenBoxGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("Zen Box")
        # Force window to front
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.lift()
        self.window.focus_force()
        
        self.final_score = 0
        self.canvas = tk.Canvas(self.window, width=400, height=300, bg='#1F1F1F')
        self.canvas.pack(pady=10)
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FED766', '#F0EAD6']
        self.box = self.canvas.create_rectangle(10, 125, 60, 175, fill=random.choice(self.colors), outline="")
        self.x_velocity = 1.5
        self.time_left = 60
        self.time_label = tk.Label(self.window, text=f"Time for calm: {self.time_left}s", font=('Arial', 10))
        self.time_label.pack(pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.move_box()
        self.update_timer()
        
    def move_box(self):
        try:
            self.canvas.move(self.box, self.x_velocity, 0)
            pos = self.canvas.coords(self.box)
            if pos[2] >= 400 or pos[0] <= 0:
                self.x_velocity *= -1
            if self.time_left > 0:
                self.window.after(20, self.move_box)
        except:
            pass
            
    def on_canvas_click(self, event):
        self.canvas.itemconfig(self.box, fill=random.choice(self.colors))
        
    def update_timer(self):
        try:
            if self.time_left > 0:
                self.time_left -= 1
                self.time_label.config(text=f"Time for calm: {self.time_left}s")
                self.window.after(1000, self.update_timer)
            else:
                self.final_score = 1
                self.window.destroy()
        except:
            pass


class DrawingCanvasGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("Drawing Canvas")
        # Force window to front
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.lift()
        self.window.focus_force()
        
        self.final_score = 0
        self.current_color = "black"
        controls_frame = tk.Frame(self.window)
        controls_frame.pack(pady=5)
        colors = ["black", "red", "green", "blue", "yellow", "orange", "purple"]
        for color in colors:
            tk.Button(controls_frame, bg=color, width=2, command=lambda c=color: self.set_color(c)).pack(side="left", padx=2)
        self.canvas = tk.Canvas(self.window, width=500, height=400, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        tk.Button(self.window, text="Clear", command=lambda: self.canvas.delete("all")).pack(side="left", padx=10, pady=5)
        tk.Button(self.window, text="Finish", command=self.end_game).pack(side="right", padx=10, pady=5)
        
    def set_color(self, color):
        self.current_color = color
        
    def draw(self, event):
        self.canvas.create_oval(event.x-2, event.y-2, event.x+2, event.y+2, fill=self.current_color, outline=self.current_color)
        
    def end_game(self):
        self.final_score = 1
        self.window.destroy()


class TargetPracticeGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("Target Practice")
        # Force window to front
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.lift()
        self.window.focus_force()
        
        self.final_score = 0
        self.canvas = tk.Canvas(self.window, width=500, height=400, bg='black')
        self.canvas.pack()
        self.score, self.time_left, self.combo = 0, 30, 0
        self.score_label = tk.Label(self.window, text=f"Score: {self.score}", font=('Arial', 14))
        self.score_label.pack(side="left", padx=10)
        self.combo_label = tk.Label(self.window, text=f"Combo: x{self.combo}", font=('Arial', 14))
        self.combo_label.pack(side="left", padx=10)
        self.time_label = tk.Label(self.window, text=f"Time: {self.time_left}", font=('Arial', 14))
        self.time_label.pack(side="right", padx=10)
        self.canvas.bind("<Button-1>", self.on_miss)
        self.spawn_target()
        self.update_timer()
        
    def spawn_target(self):
        self.canvas.delete("all")
        x, y = random.randint(30, 470), random.randint(30, 370)
        self.target_radius = 30
        self.target = self.canvas.create_oval(x-self.target_radius, y-self.target_radius, x+self.target_radius, y+self.target_radius, fill='red', tags="target")
        self.canvas.tag_bind("target", "<Button-1>", self.on_target_click)
        self.shrink_target()
        
    def shrink_target(self):
        try:
            if self.time_left > 0 and self.target_radius > 5:
                self.target_radius -= 0.5
                coords = self.canvas.coords(self.target)
                cx, cy = (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
                self.canvas.coords(self.target, cx-self.target_radius, cy-self.target_radius, cx+self.target_radius, cy+self.target_radius)
                self.window.after(50, self.shrink_target)
            elif self.time_left > 0:
                self.on_miss(None)
        except:
            pass
            
    def on_target_click(self, event):
        self.combo += 1
        points = int(35 - self.target_radius) * self.combo
        self.score += points
        self.score_label.config(text=f"Score: {self.score}")
        self.combo_label.config(text=f"Combo: x{self.combo}")
        self.spawn_target()
        
    def on_miss(self, event):
        self.combo = 0
        self.score -= 5
        self.score_label.config(text=f"Score: {self.score}")
        self.combo_label.config(text=f"Combo: x{self.combo}")
        if self.time_left > 0:
            self.spawn_target()
            
    def update_timer(self):
        try:
            if self.time_left > 0:
                self.time_left -= 1
                self.time_label.config(text=f"Time: {self.time_left}")
                self.window.after(1000, self.update_timer)
            else:
                self.end_game()
        except:
            pass
            
    def end_game(self):
        self.final_score = self.score
        self.window.destroy()


class TypingSpeedGame:
    def __init__(self, master):
        self.window = tk.Toplevel(master)
        self.window.title("Typing Speed Test")
        # Force window to front
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.lift()
        self.window.focus_force()
        
        self.final_score = 0
        self.words = ["python", "focus", "challenge", "cognitive", "model", "develop", "accuracy", "breathe", "relax"]
        self.time_left, self.score, self.total_chars = 45, 0, 0
        tk.Label(self.window, text=f"Type as many words as you can in {self.time_left} seconds!", font=('Arial', 14)).pack(pady=10)
        self.time_label = tk.Label(self.window, text=f"Time: {self.time_left}", font=('Arial', 14))
        self.time_label.pack()
        self.score_label = tk.Label(self.window, text="Score: 0", font=('Arial', 14))
        self.score_label.pack()
        self.word_label = tk.Label(self.window, text="", font=('Arial', 24, 'bold'), fg='blue')
        self.word_label.pack(pady=10)
        self.entry = tk.Entry(self.window, font=('Arial', 18), justify='center')
        self.entry.pack(pady=10, padx=10)
        self.entry.bind("<Return>", self.check_word)
        self.entry.focus_set()
        self.next_word()
        self.update_timer()
        
    def next_word(self):
        self.entry.delete(0, tk.END)
        self.current_word = random.choice(self.words)
        self.word_label.config(text=self.current_word)
        
    def check_word(self, event):
        if self.time_left > 0 and self.entry.get() == self.current_word:
            self.score += 1
            self.total_chars += len(self.current_word)
            self.score_label.config(text=f"Score: {self.score}")
            self.next_word()
            
    def update_timer(self):
        try:
            if self.time_left > 0:
                self.time_left -= 1
                self.time_label.config(text=f"Time: {self.time_left}")
                self.window.after(1000, self.update_timer)
            else:
                self.end_game()
        except:
            pass
            
    def end_game(self):
        self.entry.config(state='disabled')
        wpm = (self.total_chars / 5) / (45 / 60) if self.total_chars > 0 else 0
        self.final_score = wpm
        self.window.destroy()


class GameAnalysisWindow:
    def __init__(self, master, state, game_class_name, score):
        self.window = tk.Toplevel(master)
        self.window.title("Game Report")
        # Force window to front
        self.window.attributes('-topmost', True)
        self.window.after(100, lambda: self.window.attributes('-topmost', False))
        self.window.lift()
        self.window.focus_force()
        
        self.window.config(padx=30, pady=20, bg="#F0F0F0")
        
        tk.Label(self.window, text="Game Performance Report", font=('Arial', 18, 'bold'), bg="#F0F0F0").pack(pady=(0, 20))
        
        score_text, insight, takeaway = self.get_analysis_message(state, game_class_name, score)
        
        tk.Label(self.window, text="Your Performance:", font=('Arial', 12, 'bold'), bg="#F0F0F0").pack(anchor="w")
        tk.Label(self.window, text=score_text, font=('Arial', 12), bg="#F0F0F0").pack(anchor="w", pady=(0, 10))
        
        tk.Label(self.window, text="Our Insight:", font=('Arial', 12, 'bold'), bg="#F0F0F0").pack(anchor="w")
        tk.Label(self.window, text=insight, font=('Arial', 12, 'italic'), wraplength=400, justify="left", bg="#F0F0F0").pack(anchor="w", pady=(0, 10))
        
        tk.Label(self.window, text="Your Takeaway:", font=('Arial', 12, 'bold'), bg="#F0F0F0").pack(anchor="w")
        tk.Label(self.window, text=takeaway, font=('Arial', 12), wraplength=400, justify="left", bg="#F0F0F0").pack(anchor="w")
        
        tk.Button(self.window, text="Finish Session", command=self.finish_session, font=('Arial', 12, 'bold'), bg="#4CAF50", fg="white", relief="flat").pack(pady=20)
        
    def finish_session(self):
        self.window.destroy()
        
    def get_analysis_message(self, state, game_class_name, score):
        if game_class_name in ['ZenBoxGame', 'DrawingCanvasGame']:
            score_text = "Session Completed"
            insight = f"You took a moment for a calming activity, which is a powerful way to respond to feelings of {state}."
            takeaway = "Remember this feeling of calm. If you feel overwhelmed later, try to recall this moment or take 60 seconds to focus on your breath."
            return score_text, insight, takeaway
        
        elif game_class_name == 'TargetPracticeGame':
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
        
        elif game_class_name == 'TypingSpeedGame':
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


# --- GAME LIBRARY ---

GAME_LIBRARY = {
    'stress': ['ZenBoxGame', 'DrawingCanvasGame'],
    'relax': ['DrawingCanvasGame', 'ZenBoxGame'],
    'focus': ['TargetPracticeGame', 'TypingSpeedGame']
}

GAME_CLASSES = {
    'ZenBoxGame': ZenBoxGame,
    'DrawingCanvasGame': DrawingCanvasGame,
    'TargetPracticeGame': TargetPracticeGame,
    'TypingSpeedGame': TypingSpeedGame
}


# --- GAME SEQUENCE FUNCTION ---

def run_game_sequence(state, index, game_flag):
    """Runs in separate process"""
    try:
        root = tk.Tk()
        root.withdraw()
        
        game_class_name = GAME_LIBRARY[state][index]
        game_class = GAME_CLASSES[game_class_name]
        
        game_instance = game_class(root)
        root.wait_window(game_instance.window)
        
        analysis_window = GameAnalysisWindow(root, state, game_class_name, game_instance.final_score)
        root.wait_window(analysis_window.window)
        
        root.destroy()
        
    except Exception as e:
        print(f"Error in game sequence: {e}")
    finally:
        game_flag.value = 0


# --- FLASK API ---

@app.route('/api/start-game', methods=['POST'])
def start_game():
    if game_running.value == 1:
        return jsonify({'success': False, 'error': 'A game is already running'}), 400
    
    try:
        data = request.json
        state = data.get('state')
        index = data.get('index')
        
        if state not in GAME_LIBRARY:
            return jsonify({'success': False, 'error': 'Invalid state'}), 400
        
        if index not in [0, 1]:
            return jsonify({'success': False, 'error': 'Invalid game index'}), 400
        
        game_running.value = 1
        
        game_process = multiprocessing.Process(
            target=run_game_sequence, 
            args=(state, index, game_running)
        )
        game_process.start()
        
        return jsonify({'success': True, 'message': 'Game started successfully'})
        
    except Exception as e:
        game_running.value = 0
        return jsonify({'success': False, 'error': str(e)}), 500


# CRITICAL: Must have this guard for Windows multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)
