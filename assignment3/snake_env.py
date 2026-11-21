"""Snake on a small board, for tabular Q-Learning.

The board has SIZE x SIZE cells and everything outside it is wall. Positions are
(x, y): x grows to the right and y grows downwards, so "up" means a smaller y.

    env = SnakeEnv(seed=0)
    state = env.reset()
    next_state, reward, done, info = env.step(STRAIGHT)

A state is a tuple of 11 integers, each 0 or 1, in the order of STATE_FEATURES.
"""
import random

SIZE = 8
START_LENGTH = 3
MAX_STEPS_WITHOUT_FOOD = 40

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_TIMEOUT = -5.0
REWARD_STEP = -0.1

STRAIGHT, TURN_LEFT, TURN_RIGHT = 0, 1, 2
ACTIONS = (STRAIGHT, TURN_LEFT, TURN_RIGHT)
ACTION_NAMES = ("straight", "turn_left", "turn_right")

UP, RIGHT, DOWN, LEFT = (0, -1), (1, 0), (0, 1), (-1, 0)
CLOCKWISE = (UP, RIGHT, DOWN, LEFT)
DIRECTION_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}

STATE_FEATURES = (
    "danger_straight", "danger_left", "danger_right",
    "moving_up", "moving_down", "moving_left", "moving_right",
    "food_up", "food_down", "food_left", "food_right",
)


class SnakeEnv:
    """One game of Snake. Call reset() to start an episode and step(action) to play it.

    Attributes you may read (do not change them):
        snake       list of (x, y) cells, head first
        food        (x, y) of the food, or None if the board is full
        direction   the direction of the last move, one of UP, RIGHT, DOWN, LEFT
        food_eaten  food eaten in this episode
        steps       steps played in this episode
        end_reason  None while playing, then "wall", "body", "timeout" or "full"
    """

    def __init__(self, size=SIZE, max_steps_without_food=MAX_STEPS_WITHOUT_FOOD, seed=None):
        if size < START_LENGTH + 2:
            raise ValueError(f"size must be at least {START_LENGTH + 2}")
        self.size = size
        self.max_steps_without_food = max_steps_without_food
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        """Start a new episode and return its first state."""
        middle = self.size // 2
        self.direction = RIGHT
        self.snake = [(middle - i, middle) for i in range(START_LENGTH)]
        self.food = self._random_free_cell()
        self.food_eaten = 0
        self.steps = 0
        self.steps_without_food = 0
        self.total_reward = 0.0
        self.end_reason = None
        return self.get_state()

    def step(self, action):
        """Play one move and return (next_state, reward, done, info).

        info is a dict with "ate" (True if the snake ate on this step) and
        "reason" (None, or why the episode ended).
        """
        if self.end_reason is not None:
            raise RuntimeError("The episode is over: call reset() first.")
        if action not in ACTIONS:
            raise ValueError(f"action must be 0, 1 or 2, got {action!r}")
        self.direction = self._turned(int(action))
        head = self._ahead(self.direction)
        self.steps += 1
        ate = False

        if self._blocked(head):
            self.end_reason = "wall" if self._outside(head) else "body"
            reward = REWARD_DEATH
        else:
            self.snake.insert(0, head)
            if head == self.food:
                ate = True
                self.food_eaten += 1
                self.steps_without_food = 0
                reward = REWARD_FOOD
                self.food = self._random_free_cell()
                if self.food is None:
                    self.end_reason = "full"
            else:
                self.snake.pop()
                self.steps_without_food += 1
                reward = REWARD_STEP
                if self.steps_without_food >= self.max_steps_without_food:
                    self.end_reason = "timeout"
                    reward = REWARD_TIMEOUT

        self.total_reward += reward
        done = self.end_reason is not None
        return self.get_state(), reward, done, {"ate": ate, "reason": self.end_reason}

    def get_state(self):
        """The 11 binary features, in the order of STATE_FEATURES."""
        index = CLOCKWISE.index(self.direction)
        left, right = CLOCKWISE[index - 1], CLOCKWISE[(index + 1) % 4]
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food if self.food is not None else (head_x, head_y)
        features = (
            self._blocked(self._ahead(self.direction)),
            self._blocked(self._ahead(left)),
            self._blocked(self._ahead(right)),
            self.direction == UP,
            self.direction == DOWN,
            self.direction == LEFT,
            self.direction == RIGHT,
            food_y < head_y,
            food_y > head_y,
            food_x < head_x,
            food_x > head_x,
        )
        return tuple(int(value) for value in features)

    def render(self):
        """The board as text: O is the head, o the body, * the food and # the wall."""
        rows = ["#" * (self.size + 2)]
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                cell = (x, y)
                if cell == self.snake[0]:
                    row += "O"
                elif cell in self.snake:
                    row += "o"
                elif cell == self.food:
                    row += "*"
                else:
                    row += "."
            rows.append("#" + row + "#")
        rows.append("#" * (self.size + 2))
        return "\n".join(rows)

    def _turned(self, action):
        index = CLOCKWISE.index(self.direction)
        if action == TURN_LEFT:
            return CLOCKWISE[index - 1]
        if action == TURN_RIGHT:
            return CLOCKWISE[(index + 1) % 4]
        return self.direction

    def _ahead(self, direction):
        head_x, head_y = self.snake[0]
        return head_x + direction[0], head_y + direction[1]

    def _outside(self, cell):
        return not (0 <= cell[0] < self.size and 0 <= cell[1] < self.size)

    def _blocked(self, cell):
        # The tail moves away on the same step, so moving into it is safe.
        return self._outside(cell) or cell in self.snake[:-1]

    def _random_free_cell(self):
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied]
        return self.rng.choice(free) if free else None
