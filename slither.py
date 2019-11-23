from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import (
    presence_of_element_located,
    element_to_be_clickable,
    presence_of_all_elements_located,
)

from constants import SLITHERIO_URL

play_button_classname = "nsi"
play_button_class_index = 2

# This example requires Selenium WebDriver 3.13 or newer
with webdriver.Chrome() as driver:
    wait = WebDriverWait(driver, 15)
    driver.get(SLITHERIO_URL)

    element = wait.until(
        # element_to_be_clickable((By.CLASS_NAME, play_button_classname))
        # presence_of_element_located((By.CLASS_NAME, play_button_classname))
        presence_of_all_elements_located((By.CLASS_NAME, play_button_classname))
    )[play_button_class_index]
    element.click()

    # This will be useful at some point
    # while True:
    #     res = driver.execute_script("return playing;")
    #     print(res)
    #     if res:
    #         break

    while True:
        res = driver.execute_script("return (typeof score !== 'undefined')")
        if res:
            break

    while True:
        score = driver.execute_script("return score;")
        print(score)
