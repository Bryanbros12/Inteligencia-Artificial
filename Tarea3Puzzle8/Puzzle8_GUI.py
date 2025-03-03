import tkinter as tk
from queue import Queue
import random

# Estado objetivo
GOAL_STATE = "123456780"
MOVES = {
    "Up": -3,
    "Down": 3,
    "Left": -1,
    "Right": 1
}


class Puzzle8:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle")
        self.state = self.shuffle()
        self.move_count = 0

        self.buttons = []
        self.create_widgets()

    def shuffle(self):
        nums = list("123456780")
        random.shuffle(nums)
        return "".join(nums)

    def create_widgets(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        for i in range(9):
            btn = tk.Button(self.frame, text=self.state[i], font=("Arial", 24), height=2, width=5)
            btn.grid(row=i // 3, column=i % 3)
            self.buttons.append(btn)

        self.label_moves = tk.Label(self.root, text=f"Movimientos: {self.move_count}", font=("Arial", 14))
        self.label_moves.pack()

        self.btn_manual = tk.Button(self.root, text="Jugar Manual", command=self.enable_manual)
        self.btn_manual.pack()

        self.btn_solve = tk.Button(self.root, text="Resolver con IA", command=self.solve_with_ai)
        self.btn_solve.pack()

        self.btn_reset = tk.Button(self.root, text="Reiniciar", command=self.reset_game)
        self.btn_reset.pack()

    def enable_manual(self):
        self.btn_manual.config(state=tk.DISABLED)
        self.btn_solve.config(state=tk.DISABLED)
        self.root.bind("<Up>", lambda e: self.move("Up"))
        self.root.bind("<Down>", lambda e: self.move("Down"))
        self.root.bind("<Left>", lambda e: self.move("Left"))
        self.root.bind("<Right>", lambda e: self.move("Right"))

    def move(self, direction):
        index = self.state.index("0")
        new_index = index + MOVES[direction]
        if 0 <= new_index < 9 and not (index % 3 == 2 and direction == "Right") and not (
                index % 3 == 0 and direction == "Left"):
            self.state = list(self.state)
            self.state[index], self.state[new_index] = self.state[new_index], self.state[index]
            self.state = "".join(self.state)
            self.move_count += 1
            self.update_ui()

    def update_ui(self):
        for i in range(9):
            self.buttons[i].config(text=self.state[i] if self.state[i] != "0" else "")
        self.label_moves.config(text=f"Movimientos: {self.move_count}")

    def solve_with_ai(self):
        self.btn_manual.config(state=tk.DISABLED)
        self.btn_solve.config(state=tk.DISABLED)
        queue = Queue()
        queue.put((self.state, []))
        visited = set()

        while not queue.empty():
            current, path = queue.get()
            if current == GOAL_STATE:
                self.animate_solution(path)
                return
            visited.add(current)

            index = current.index("0")
            for direction, move in MOVES.items():
                new_index = index + move
                if 0 <= new_index < 9 and not (index % 3 == 2 and direction == "Right") and not (
                        index % 3 == 0 and direction == "Left"):
                    new_state = list(current)
                    new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
                    new_state = "".join(new_state)
                    if new_state not in visited:
                        queue.put((new_state, path + [direction]))

    def animate_solution(self, path):
        if not path:
            return
        self.move(path.pop(0))
        self.root.after(500, lambda: self.animate_solution(path))

    def reset_game(self):
        self.state = self.shuffle()
        self.move_count = 0
        self.update_ui()
        self.btn_manual.config(state=tk.NORMAL)
        self.btn_solve.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = Puzzle8(root)
    root.mainloop()
