# Training a Neural Network to play [Slither.io](http://slither.io) using Reinforcement Learning

## Requirements
* [Selenium chromedriver](https://sites.google.com/a/chromium.org/chromedriver/downloads)
* Tensorflow (may need to roll back from 2.0.0 to 2.0.0-beta, due to keras-rl reqs)


## Resources
* [Not working example but has some reference code](https://botfather.io/docs/wizard/simple-agario-bot-tutorial/)
* [Example use of Selenium](https://automatetheboringstuff.com/chapter11/)
* [Specific part of Selenium documentation, highlighting important functions](https://selenium.dev/selenium/docs/api/py/webdriver_remote/selenium.webdriver.remote.webdriver.html?highlight=get_screenshot#selenium.webdriver.remote.webdriver.WebDriver.get_screenshot_as_file)
* [Agario driver that uses selenium](https://github.com/gsgalloway/agar-io-driver)
* [Article on DDQNs](https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/)
* [Actor-Critic](https://sergioskar.github.io/Actor_critics/)
* [Repo with an implementation of A3C](https://github.com/germain-hug/Deep-RL-Keras)


## TODO
* [x] Add action handling in env.step
* [x] Add image processing to observation (convert to np.array)
* [x] Find A3C implementation to use
* [ ] Use env with A3C model
