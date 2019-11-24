from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import (
    presence_of_element_located,
    element_to_be_clickable,
)

#
# from PIL import Image

from selenium.webdriver.chrome.options import Options

chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("window-size=300,300")

# This example requires Selenium WebDriver 3.13 or newer
with webdriver.Chrome(options=chrome_options) as driver:
    wait = WebDriverWait(driver, 15)
    driver.get("https://agar.io")
    first_result = wait.until(element_to_be_clickable((By.ID, "play")))
    first_result.click()

    driver.get_screenshot_as_file("yeet.png")

    # a = driver.get_screenshot_as_png()
    # Use PIL to open this as an image, then convert to np array

    score_element = wait.until(presence_of_element_located((By.ID, "score")))
    while True:
        score = score_element.get_attribute("innerHTML")
