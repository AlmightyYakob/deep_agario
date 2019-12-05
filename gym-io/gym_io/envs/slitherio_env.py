import math
import numpy as np

from gym import Env, spaces

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.expected_conditions import (
    # presence_of_element_located,
    # element_to_be_clickable,
    presence_of_all_elements_located,
)


SLITHERIO_URL = "https://slither.io"
SLITHERIO_CONNECTION_TIMEOUT_SECONDS = 15
SLITHERIO_INITIAL_LENGTH = 10

PLAY_BUTTON_CLASSNAME = "nsi"
PLAY_BUTTON_CLASS_INDEX = 2

OVERLAY_CSS_SELECTOR = "div.nsi"
OVERLAY_INDICES = [12, 13, 14, 15, 16, 17]

JS_MOUSE_VAR = "window.mouse"

# Num degrees + the boost action
DEGREE_GRANULARITY = 360
NUM_ACTIONS = DEGREE_GRANULARITY + 1
MOUSE_RADIUS_FRACTION = 0.1

DEFAULT_CHROME_OPTIONS = {"width": 300, "height": 300, "headless": True}


class SlitherIOEnv(Env):
    metadata = {"render.modes": ["human"]}

    # TBD, need to figure out full observation space
    # observation_space =
    # observation_space.shape =

    # Representing degrees to move mouse, might change
    action_space = spaces.Discrete(NUM_ACTIONS)
    action_space.shape = (1, NUM_ACTIONS)

    def __init__(self, **kwargs):
        options = {**DEFAULT_CHROME_OPTIONS, **kwargs}

        desired_window_width = options["width"]
        desired_window_height = options["height"]
        headless = options["headless"]

        chrome_options = Options()
        chrome_options.add_argument(
            f"window-size={desired_window_width},{desired_window_height}"
        )

        if headless:
            chrome_options.add_argument("--headless")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.actions = ActionChains(self.driver)
        self.wait = WebDriverWait(self.driver, 15)

        # Initialize connection
        self.driver.get(SLITHERIO_URL)

        size = self.driver.get_window_size()
        self.window_size = (size["width"], size["height"])
        self.window_center = (self.window_size[0] / 2, self.window_size[1] / 2)
        self.mouse_radius = MOUSE_RADIUS_FRACTION * min(self.window_size)
        print("radius = ", self.mouse_radius)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*self.window_size, 1), dtype=np.uint8
        )

        self.inject_mouse_tracking_script()
        self.center_mouse()

    def reset_game(self):
        self.length = SLITHERIO_INITIAL_LENGTH
        self.playing = False
        self.snake_exists = False
        self.game_ready = False

    def inject_mouse_tracking_script(self):
        self.driver.execute_script(
            f"let currentFunc = onmousemove;"
            f"{JS_MOUSE_VAR} = {{x: 0, y: 0}};"
            f"onmousemove = function(e){{ currentFunc(e); {JS_MOUSE_VAR}.x = e.clientX; {JS_MOUSE_VAR}.y = e.clientY }};"
        )
        self.actions.move_by_offset(1, 1).perform()

    def center_mouse(self):
        pos = self.get_mouse_pos()
        while (pos != self.window_center):
            print(pos)
            pos = self.get_mouse_pos()
            offset = (self.window_center[0] - pos[0], self.window_center[1] - pos[1])
            self.actions.move_by_offset(*offset).perform()

    def get_mouse_pos(self):
        pos = self.driver.execute_script(f"return {JS_MOUSE_VAR};")
        return (pos["x"], pos["y"])

    def wait_for_existing_values(self):
        if not self.playing:
            play_button_element = self.wait.until(
                presence_of_all_elements_located((By.CLASS_NAME, PLAY_BUTTON_CLASSNAME))
            )[PLAY_BUTTON_CLASS_INDEX]
            play_button_element.click()

            self.playing = True

        while not self.snake_exists:
            if self.driver.execute_script("return snake;"):
                self.snake_exists = True

        self.game_ready = True

    def hide_overlay(self):
        self.driver.execute_script(
            "var elements = document.querySelectorAll(arguments[0]);"
            "arguments[1].forEach(i => elements[i].style.display = 'none');",
            OVERLAY_CSS_SELECTOR,
            OVERLAY_INDICES,
        )

    def get_score(self):
        script = (
            "return (Math.floor(15 * (fpsls[snake.sct]"
            "+ snake.fam / fmlts[snake.sct] - 1) - 5) / 1)"
        )
        score = self.driver.execute_script(script)
        return score

    def observe(self):
        shot = self.driver.get_screenshot_as_png()
        return shot

    def step(self, action):
        if not self.game_ready:
            self.wait_for_existing_values()

        # obs = self.observe()
        obs = 1

        score = self.get_score()
        # print("Score:", score)

        # Take action
        if action == self.action_space.n:
            # Special action
            pass
        else:
            radians = math.radians(action)
            print("action = ", action)
            print("radians = ", radians)
            print("cos = ", math.cos(radians))
            print("sin = ", math.sin(radians))

            target = (
                self.window_center[0] + math.cos(radians) * self.mouse_radius,
                self.window_center[1] + math.sin(radians) * self.mouse_radius,
            )

            mouse_pos = self.get_mouse_pos()

            offset = (target[0] - mouse_pos[0], target[1] - mouse_pos[1])
            offset = (offset[0]/10, offset[1]/10)

            print("WINDOW SIZE:", self.window_size)
            print("POS:", mouse_pos)
            print("TARGET:", target)
            print("OFFSET:", offset)
            print("MATH:", (offset[0] + mouse_pos[0], offset[1] + mouse_pos[1]))
            self.actions.move_by_offset(*offset).perform()

            # self.actions.move_by_offset(target[0]/100, target[1]/100).perform()
            # self.driver.
            # print(new_mouse_x, new_mouse_y)

        # new_score = self.get_score()
        # reward = new_score - score
        # self.length = new_score

        # Placeholder
        reward = 1
        return (obs, reward, False, {})
        # action is a
        # screenshot with selenium              (state)
        # Get current score = r0 with selenium
        # Take provided action with selenium    (action)
        # Get current score = r1 with selenium
        # Calculate reward = r1 - r0            (reward)
        # screenshot with selenium              (next state)

    def reset(self):
        self.reset_game()
        # Observe

    def render(self, mode="human", close=False):
        # Probably ignore for now
        pass
