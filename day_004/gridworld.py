import numpy as np

from typing import Tuple
from PIL import Image, ImageDraw
from environment import Environment
from value_iteration import value_iteration_loop

Arr = np.ndarray

class Norvig(Environment):
    def dynamics(self, state: int, action: int) -> Tuple[Arr, Arr, Arr]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))
        out_probs = np.zeros(self.num_actions) + 0.1
        out_probs[action] = 0.7
        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]
        for (i, s_new) in enumerate(new_states):
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue
            new_state = state_index(s_new)
            if new_state in self.walls:
                out_states[i] = state
            else:
                out_states[i] = new_state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]
        return (out_states, out_rewards, out_probs)

    def render(self, pi: Arr, filename: str):
        assert len(pi) == self.num_states
        emoji = ["⬆️", "➡️", "⬇️", "⬅️"]
        grid = [emoji[act] for act in pi]
        grid[3] = "🟩"
        grid[7] = "🟥"
        grid[11] = "⬛"
        grid[9] = "⬛"
        grid[24] = "⬛"
        grid[23] = "⬛"
        
        print("  ".join(grid[0:5]) + "\n" + "  ".join(grid[5:10]) + "\n" + "  ".join(grid[10:15]) + "\n" + "  ".join(grid[15:20]) + "\n" + "  ".join(grid[20:]))

        # create file for the policy iteration
        cell_size = 100
        cols = 5
        rows = 5
        img = Image.new('RGB', (cell_size * cols, cell_size * rows), 'white')
        draw = ImageDraw.Draw(img)
        
        colors = {
            "🟩": (0, 255, 0),
            "🟥": (255, 0, 0),
            "⬛": (0, 0, 0)
        }
        
        def draw_arrow(direction, x, y, size):
            center_x = x + size // 2
            center_y = y + size // 2
            arrow_size = size // 3
            if direction == "⬆️":
                points = [
                    (center_x, center_y - arrow_size),
                    (center_x - arrow_size, center_y),
                    (center_x + arrow_size, center_y)
                ]
            elif direction == "➡️":
                points = [
                    (center_x + arrow_size, center_y),
                    (center_x, center_y - arrow_size),
                    (center_x, center_y + arrow_size)
                ]
            elif direction == "⬇️":
                points = [
                    (center_x, center_y + arrow_size),
                    (center_x - arrow_size, center_y),
                    (center_x + arrow_size, center_y)
                ]
            elif direction == "⬅️":
                points = [
                    (center_x - arrow_size, center_y),
                    (center_x, center_y - arrow_size),
                    (center_x, center_y + arrow_size)
                ]
            else:
                return
            draw.polygon(points, fill=(0, 0, 0))
        
        for i in range(len(grid)):
            row = i // cols
            col = i % cols
            x = col * cell_size
            y = row * cell_size
            emoji_char = grid[i]
            if emoji_char in colors:
                draw.rectangle([x, y, x + cell_size, y + cell_size], fill=colors[emoji_char])
            else:
                draw.rectangle([x, y, x + cell_size, y + cell_size], fill='white')
                draw_arrow(emoji_char, x, y, cell_size)
        
        img.save(filename)

    def __init__(self, penalty=-0.04):
        self.height = 5
        self.width = 5
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        self.dim = (self.height, self.width)
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([11, 9, 24, 23], dtype=int)
        self.goal_rewards = np.array([1.0, -1])
        super().__init__(num_states, num_actions, start=8, terminal=terminal)


if __name__ == "__main__":
    penalty = -0.04
    norvig = Norvig(penalty)
    pi_opt = value_iteration_loop(norvig, gamma=0.99)
    norvig.render(pi_opt, "optimal_policy.png")