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

from PIL import Image
from io import BytesIO
import cv2


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
# NUM_ACTIONS = DEGREE_GRANULARITY + 1
NUM_ACTIONS = DEGREE_GRANULARITY

MOUSE_RADIUS_FRACTION = 0.25

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

    def reset_game(self):
        self.length = SLITHERIO_INITIAL_LENGTH
        self.playing = False
        self.snake_exists = False
        self.game_ready = False

    def get_playing_status(self):
        return self.driver.execute_script("return playing;")

    def snake_is_dead(self):
        res = int(self.driver.execute_script("return snake.dead_amt;"))
        return bool(res)

    def set_mouse_pos(self, pos):
        self.driver.execute_script(f"xm = {int(pos[0])}; ym = {int(pos[1])};")

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

        # Also try with Image.frombytes
        # image = Image.frombytes(mode, size, shot)
        image = Image.open(BytesIO(shot))

        np_image = np.array(image)
        grayscale = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        return grayscale

    def take_action(self, action):
        radians = math.radians(action)
        target = (
            self.window_center[0] + math.cos(radians) * self.mouse_radius,
            self.window_center[1] + math.sin(radians) * self.mouse_radius,
        )

        offset = (
            target[0] - self.window_center[0],
            target[1] - self.window_center[1],
        )

        self.set_mouse_pos(offset)

    def step(self, action):
        """
        Move the snake in the direction represented by action

        observation - Screenshot with selenium
        reward      - The change in reward after taken action
        """

        if not self.game_ready:
            self.wait_for_existing_values()

        if not self.get_playing_status() or self.snake_is_dead():
            # Dead before any action actually taken, so neutral reward
            return (self.observe(), 0, True, {})

        score = self.get_score()

        # # Special action, currently ignore
        # if action == (self.action_space.n - 1):
        #     # self.actions.click()
        #     pass
        # else:

        self.take_action(action)

        # Take screenshot
        obs = self.observe()

        done = False
        if self.snake_is_dead():
            reward = -score
            done = True
        else:
            new_score = self.get_score()
            reward = new_score - score

        print(reward)
        return (obs, reward, done, {})

    def reset(self):
        self.reset_game()
        # Observe

    def render(self, mode="human", close=False):
        # Probably ignore for now
        pass
