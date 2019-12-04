from gym import Env, spaces

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.expected_conditions import (
    presence_of_element_located,
    # element_to_be_clickable,
    presence_of_all_elements_located,
)


SLITHERIO_URL = "https://slither.io"
SLITHERIO_CONNECTION_TIMEOUT_SECONDS = 15
SLITHERIO_INITIAL_LENGTH = 10

PLAY_BUTTON_CLASSNAME = "nsi"
PLAY_BUTTON_CLASS_INDEX = 2

LENGTH_CSS_SELECTOR = "div.nsi > span > span:nth-child(2)"
CANVAS_SELCTOR = "canvas.nsi"

OVERLAY_CSS_SELECTOR = "div.nsi"
OVERLAY_INDICES = [12, 13, 14, 15, 16, 17]


class SlitherIOEnv(Env):
    metadata = {"render.modes": ["human"]}

    # TBD, need to figure out full observation space
    # observation_space =
    # observation_space.shape =

    # Representing degrees to move mouse, might change
    action_space = spaces.Discrete(360)
    action_space.shape = (1, 360)

    def __init__(self):
        # Initialize selenium, open website but don't click play yet
        # Apply required CSS to dom elements to hide overlays and stuff

        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        # chrome_options.add_argument("window-size=300,300")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 15)

        self.driver.get(SLITHERIO_URL)

    def reset_game(self):
        self.length = SLITHERIO_INITIAL_LENGTH
        self.playing = False

    def hide_overlay(self):
        self.driver.execute_script(
            "var elements = document.querySelectorAll(arguments[0]);"
            "arguments[1].forEach(i => elements[i].style.display = 'none');",
            OVERLAY_CSS_SELECTOR,
            OVERLAY_INDICES,
        )

    def observe(self):
        if not self.playing:
            play_button_element = self.wait.until(
                presence_of_all_elements_located((By.CLASS_NAME, PLAY_BUTTON_CLASSNAME))
            )[PLAY_BUTTON_CLASS_INDEX]
            play_button_element.click()

            self.wait.until(
                presence_of_element_located((By.CSS_SELECTOR, LENGTH_CSS_SELECTOR))
            )
            self.playing = True
            # self.hide_overlay()

        shot = self.driver.get_screenshot_as_png()
        return shot

    def step(self, action):
        obs = self.observe()
        element_text = self.driver.find_element_by_css_selector(
            LENGTH_CSS_SELECTOR
        ).text

        if not element_text:
            print("Score not found in DOM!")
            exit()

        new_length = int(element_text)
        print(new_length)

        reward = new_length - self.length
        self.length = new_length
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
