# hows my driving sf bot

This is a layer on top of the model developed in [howsmy](https://github.com/tmcw/howsmy),
mainly by [oliverproud](https://github.com/oliverproud/). It takes that model, adds a scraper
and web interface on top of it.

## The good

- CAPTCHAs are solveable and now, solved. With 98% accuracy. Very much thanks to Oliver.
- The pages are parseable and parsed.

## The bad

- etimspayments uses [Incapsula](https://www.imperva.com/), a fancy application security platform.
  When it identifies this thing as a bot, it is very hard to get around.
- I'm stalled on actually implementing a bot until Twitter approves a developer account.
