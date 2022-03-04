#!/usr/bin/env python3
"""Module that contians the function availableShips."""

import requests


def availableShips(passengerCount):
    """Checks the Swapi for ships that will fit the specified number of
    pasengers.

    Args:
        passengerCount (int): The number of passengers to fit on the ship.

    Returns: avalible_ships
        avalible_ships (list[str]): The names of the ships that can accomidate
            the specified number of passengers.
    """
    url = "https://swapi-api.hbtn.io/api/starships/?format=json"
    ships = []
    avalible_ships = []

    while url:
        r_json = requests.get(url).json()
        ships += r_json['results']
        url = r_json['next']

    for ship in ships:
        passengers = ship['passengers'].replace(",", "")
        if (passengers != "n/a" and passengers != "unknown" and
                int(passengers) >= passengerCount):
            avalible_ships.append(ship['name'])

    return avalible_ships


if __name__ == "__main__":
    ships = availableShips(4)
    for ship in ships:
        print(ship)

    # Expected output
    """
CR90 corvette
Sentinel-class landing craft
Death Star
Millennium Falcon
Executor
Rebel transport
Slave 1
Imperial shuttle
EF76 Nebulon-B escort frigate
Calamari Cruiser
Republic Cruiser
Droid control ship
Scimitar
J-type diplomatic barge
AA-9 Coruscant freighter
Republic Assault ship
Solar Sailer
Trade Federation cruiser
Theta-class T-2c shuttle
Republic attack cruiser
"""
