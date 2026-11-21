"""Helpers for the notebook: basic checks, plots, a replay of one episode and results.json."""
import functools
import json
import random

from snake_env import ACTIONS, SnakeEnv

RESULT_KEYS = ("avg_food", "avg_reward", "avg_steps", "max_food", "deaths", "timeouts")
HISTORY_KEYS = ("food", "reward", "steps", "epsilon")


def _keeps_random_state(check):
    """The checks use their own seeds and then put back the notebook's random state."""

    @functools.wraps(check)
    def wrapper(*args, **kwargs):
        saved = random.getstate()
        try:
            return check(*args, **kwargs)
        finally:
            random.setstate(saved)

    return wrapper


def _passed(task):
    print(f"{task}: basic checks passed. They do not cover everything, so test your own cases too.")


@_keeps_random_state
def check_choose_action(choose_action):
    state = (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1)
    Q = {state: [0.0, 5.0, 1.0]}
    for _ in range(20):
        action = choose_action(Q, state, 0.0)
        assert action == 1, f"with epsilon = 0 the best action is 1, but choose_action returned {action!r}"

    unseen = (1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0)
    action = choose_action(Q, unseen, 0.0)
    assert action in ACTIONS, f"choose_action must return 0, 1 or 2, got {action!r}"
    assert unseen in Q, "an unseen state must be added to Q"
    assert len(Q[unseen]) == 3 and all(value == 0 for value in Q[unseen]), "an unseen state must start with 0.0 for each action"

    chosen = {choose_action(Q, state, 1.0) for _ in range(300)}
    assert chosen == set(ACTIONS), f"with epsilon = 1 every action should appear, but only {sorted(chosen)} did"
    _passed("Task 1")


@_keeps_random_state
def check_update_q_value(update_q_value):
    state, next_state = (0,) * 11, (1,) + (0,) * 10
    Q = {state: [0.0, 0.0, 0.0], next_state: [1.0, 2.0, 0.5]}
    new = update_q_value(Q, state, 0, -0.1, next_state, False, 0.5, 0.9)
    expected = 0.5 * (-0.1 + 0.9 * 2.0)
    assert abs(Q[state][0] - expected) < 1e-9, f"Q[state][0] should be {expected:.4f}, got {Q[state][0]!r}"
    assert new is not None and abs(new - expected) < 1e-9, f"update_q_value should return the new value {expected:.4f}, got {new!r}"
    assert list(Q[next_state]) == [1.0, 2.0, 0.5], "the next state's values must not change"
    _passed("Task 2")


@_keeps_random_state
def check_train_agent(train_agent):
    random.seed(0)
    output = train_agent(SnakeEnv(seed=0), 50, 0.1, 0.9, 1.0, 0.05)
    assert isinstance(output, tuple) and len(output) == 2, "train_agent must return (Q, history)"
    Q, history = output
    assert isinstance(Q, dict) and Q, "Q must be a non-empty dict after training"
    assert isinstance(history, dict), "history must be a dict"
    for key in HISTORY_KEYS:
        assert key in history, f'history is missing "{key}"'
        assert len(history[key]) == 50, f'history["{key}"] must have one value per episode (50), got {len(history[key])}'
    epsilon = history["epsilon"]
    assert abs(epsilon[0] - 1.0) < 1e-9 and abs(epsilon[-1] - 0.05) < 1e-9, "epsilon must go from epsilon_start to epsilon_end"
    _passed("Task 3")


@_keeps_random_state
def check_evaluate_agent(evaluate_agent):
    random.seed(0)
    results = evaluate_agent(SnakeEnv(seed=0), {}, 20, 1.0)
    assert isinstance(results, dict), "evaluate_agent must return a dict"
    for key in RESULT_KEYS:
        assert key in results, f'the results are missing "{key}"'
    assert results["deaths"] + results["timeouts"] == 20, "with a random agent, every episode should end in a death or a timeout"
    assert results["avg_food"] <= results["max_food"], "avg_food cannot be larger than max_food"
    _passed("Task 4")


def plot_training(history, window=100):
    """Food per episode (moving average) and epsilon, over training."""
    import matplotlib.pyplot as plt

    food = history["food"]
    window = max(1, min(window, len(food)))
    averages = [sum(food[max(0, i - window + 1):i + 1]) / (i + 1 - max(0, i - window + 1)) for i in range(len(food))]
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.5))
    left.plot(averages)
    left.set_xlabel("Episode")
    left.set_ylabel(f"Food eaten (average of last {window})")
    right.plot(history["epsilon"], color="tab:orange")
    right.set_xlabel("Episode")
    right.set_ylabel("Epsilon")
    figure.tight_layout()
    plt.show()


def show_comparison(trained, random_agent):
    """Print the trained agent and the random agent side by side."""
    print(f"{'':12}{'TRAINED':>10}{'RANDOM':>10}")
    for key in RESULT_KEYS:
        if key.startswith("avg_"):
            print(f"{key:12}{trained[key]:>10.2f}{random_agent[key]:>10.2f}")
        else:
            print(f"{key:12}{int(trained[key]):>10}{int(random_agent[key]):>10}")


def watch_episode(Q, choose_action, seed=None, max_steps=300):
    """Replay one greedy episode (epsilon = 0) as an animation."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display

    env = SnakeEnv(seed=seed)
    state = env.reset()
    frames = [(list(env.snake), env.food, env.food_eaten)]
    done = False
    while not done and len(frames) <= max_steps:
        state, _, done, _ = env.step(choose_action(Q, state, 0.0))
        frames.append((list(env.snake), env.food, env.food_eaten))

    size = env.size
    figure, axis = plt.subplots(figsize=(3.6, 3.9), dpi=60)

    def draw(index):
        snake, food, eaten = frames[index]
        grid = [[0] * size for _ in range(size)]
        for x, y in snake[1:]:
            grid[y][x] = 1
        x, y = snake[0]
        grid[y][x] = 2
        if food is not None:
            grid[food[1]][food[0]] = 3
        axis.clear()
        axis.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(["#f4f4f4", "#5aa05a", "#1f5f1f", "#d64545"]), vmin=0, vmax=3)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xticks([i - 0.5 for i in range(1, size)], minor=True)
        axis.set_yticks([i - 0.5 for i in range(1, size)], minor=True)
        axis.grid(which="minor", color="white", linewidth=1)
        axis.tick_params(which="minor", length=0)
        axis.set_title(f"Step {index}   Food {eaten}")

    movie = animation.FuncAnimation(figure, draw, frames=len(frames), interval=120)
    plt.close(figure)
    display(HTML(movie.to_jshtml()))
    print(f"The episode ended after {env.steps} steps with {env.food_eaten} food ({env.end_reason}).")


def _rounded(results):
    return {key: round(float(results[key]), 4) if key.startswith("avg_") else int(results[key]) for key in RESULT_KEYS}


def save_results(path, episodes, alpha, gamma, epsilon_start, epsilon_end, trained, random_agent):
    """Write results.json in the format I grade."""
    for name, results in (("trained", trained), ("random_agent", random_agent)):
        missing = [key for key in RESULT_KEYS if key not in results]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
    data = {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "evaluation": _rounded(trained),
        "random_baseline": _rounded(random_agent),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    print(f"Saved {path}")
