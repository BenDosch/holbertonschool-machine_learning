#!/usr/bin/env python3
"""Script that gets information about the next upcoming SpaceX launch."""

import requests

if __name__ == "__main__":
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r_json = requests.get(url).json()
    dates = []

    for launch in r_json:
        dates.append(launch['date_local'])
    launch = r_json[dates.index(min(dates))]

    date = launch['date_local']
    name = launch['name']
    rocket = launch['rocket']
    pad = launch['launchpad']

    if rocket:
        rocket_name = requests.get(
            'https://api.spacexdata.com/v4/rockets/' + rocket).json()['name']
    if pad:
        pad_json = requests.get(
            'https://api.spacexdata.com/v4/launchpads/' + pad).json()
        pad_name = pad_json['name']
        location = pad_json['locality']

    print(name + ' (' + date + ') ' + rocket_name + ' - ' + pad_name + ' (' +
          location + ')')

    # Example output
    # Starlink 4-9 (v1.5) (2022-03-03T09:35:00-05:00) Falcon 9 -
    # KSC LC 39A (Cape Canaveral)
