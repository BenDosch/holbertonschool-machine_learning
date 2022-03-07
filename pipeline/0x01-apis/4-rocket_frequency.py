#!/usr/bin/env python3
"""Script that uses the SpaceX API to show the number of launches per
rocket."""

import requests


if __name__ == "__main__":
    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()
    rockets = {}

    for launch in launches:
        rocket = launch['rocket']
        url = 'https://api.spacexdata.com/v4/rockets/' + rocket
        r = requests.get(url).json()
        rocket = r['name']

        if rockets.get(rocket) is None:
            rockets[rocket] = 1
        else:
            rockets[rocket] += 1

    rockets_sorted = sorted(rockets.items(), key=lambda x: x[0])
    rockets_sorted = sorted(rockets_sorted, key=lambda x: x[1], reverse=True)

    for rocket in rockets_sorted:
        print(rocket[0] + ": " + str(rocket[1]))
