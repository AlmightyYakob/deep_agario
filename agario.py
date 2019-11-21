from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import (
    presence_of_element_located,
    element_to_be_clickable,
)

# This example requires Selenium WebDriver 3.13 or newer
with webdriver.Chrome() as driver:
    wait = WebDriverWait(driver, 15)
    driver.get("https://agar.io")
    first_result = wait.until(element_to_be_clickable((By.ID, "play")))
    first_result.click()

    score_element = wait.until(presence_of_element_located((By.ID, "score")))
    while True:
        score = score_element.get_attribute("innerHTML")
