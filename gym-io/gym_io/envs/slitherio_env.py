import math
import numpy as np

from gym import Env, spaces

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.expected_conditions import (
    presence_of_all_elements_located,
)

from PIL import Image
from io import BytesIO
import cv2


SLITHERIO_URL = "https://slither.io"
SLITHERIO_CONNECTION_TIMEOUT_SECONDS = 15

PLAY_BUTTON_CLASSNAME = "nsi"
PLAY_BUTTON_CLASS_INDEX = 2

OVERLAY_CSS_SELECTOR = "div.nsi"
OVERLAY_INDICES = [12, 13, 14, 15, 16, 17, 18]

DEGREE_GRANULARITY = 12
NUM_EXTRA_ACTIONS = 0

MOUSE_RADIUS_FRACTION = 0.25

OBSERVATION_INTENSITY_THRESHOLD = 255 / 4

DEFAULT_CHROME_OPTIONS = {
    "width": 300,
    "height": 300,
    "granularity": DEGREE_GRANULARITY,
    "headless": True,
}


class SlitherIOEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):
        options = {**DEFAULT_CHROME_OPTIONS, **kwargs}

        desired_window_width = options["width"]
        desired_window_height = options["height"]
        headless = options["headless"]

        chrome_options = Options()
        chrome_options.add_argument(
            f"window-size={desired_window_width},{desired_window_height}"
        )
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])

        self.action_space = spaces.Discrete(options["granularity"] + NUM_EXTRA_ACTIONS)
        self.action_space.shape = (1, self.action_space.n)

        if headless:
            chrome_options.add_argument("--headless")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.actions = ActionChains(self.driver)
        self.wait = WebDriverWait(self.driver, 15)

        # Initialize connection
        self.driver.get(SLITHERIO_URL)
        self.reset_game()

        width, height = self.get_inner_window_size()

        self.window_size = (width, height)
        self.window_center = (width / 2, height / 2)
        self.mouse_radius = MOUSE_RADIUS_FRACTION * min(self.window_size)
        print("radius = ", self.mouse_radius)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*self.window_size[::-1], 1), dtype=np.uint8
        )

    def reset_game(self):
        self.driver.refresh()
        self.playing = False
        self.overlay_hidden = False
        self.snake_exists = False
        self.game_ready = False

    def get_playing_status(self):
        return self.driver.execute_script("return playing;")

    def snake_is_dead(self):
        snake_object_is_null = bool(
            self.driver.execute_script("return snake === null;")
        )
        if snake_object_is_null:
            return True

        snake_is_dead = bool(int(self.driver.execute_script("return snake.dead_amt;")))
        return snake_is_dead

    def get_inner_window_size(self):
        return tuple(
            self.driver.execute_script(
                "return [window.innerWidth, window.innerHeight];"
            )
        )

    def set_mouse_pos(self, pos):
        self.driver.execute_script(f"xm = {int(pos[0])}; ym = {int(pos[1])};")

    def wait_until_game_ready(self):
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
        self.overlay_hidden = True

    def set_low_quality(self):
        self.driver.execute_script("high_quality = false;")

    def get_score(self):
        script = (
            "return (Math.floor(15 * (fpsls[snake.sct]"
            "+ snake.fam / fmlts[snake.sct] - 1) - 5) / 1)"
        )
        score = self.driver.execute_script(script)
        return score

    def observe(self):
        shot = self.driver.get_screenshot_as_png()
        image = Image.open(BytesIO(shot))
        np_image = np.array(image)
        grayscale = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # TODO: Finetune clipping value, or change
        # to use exact background color values
        indices = grayscale < OBSERVATION_INTENSITY_THRESHOLD
        grayscale[indices] = 0

        return grayscale

    def take_action(self, action):
        degrees = (action / self.action_space.n) * 360
        radians = math.radians(degrees)

        target = (
            self.window_center[0] + math.cos(radians) * self.mouse_radius,
            self.window_center[1] - math.sin(radians) * self.mouse_radius,
        )

        offset = (
            target[0] - self.window_center[0],
            target[1] - self.window_center[1],
        )
        self.set_mouse_pos(offset)

    def step(self, action):
        """
        Move the snake in the direction represented by action

        action      - A number representing an angle to point the snake in
        observation - Screenshot with selenium
        reward      - The change in reward after taken action
        """

        if not self.game_ready:
            self.wait_until_game_ready()
            self.hide_overlay()
            self.set_low_quality()

        if not self.get_playing_status() or self.snake_is_dead():
            # Dead before any action actually taken, so neutral reward
            return (self.observe(), 0, True, {})

        score = self.get_score()
        self.take_action(action)

        # Take screenshot
        obs = self.observe()

        done = False
        if self.snake_is_dead():
            # reward = -score
            # TODO reward should be a constant negative
            reward = -500
            done = True
            print(f"\n{'-'*10}DEAD{'-'*10}\n")
        else:
            new_score = self.get_score()
            reward = new_score - score

        return (obs, reward, done, {})

    def reset(self):
        self.reset_game()
        return self.observe()

    def render(self, mode="human", close=False):
        # Probably ignore for now
        pass

    def close(self):
        self.driver.quit()
