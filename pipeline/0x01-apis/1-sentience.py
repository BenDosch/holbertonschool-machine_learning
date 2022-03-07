#!/usr/bin/env python3
"""Module that contains the function sentientPlanets."""


import requests


def sentientPlanets():
    """A function that returns the list of names of the home planets of all
    sentient species in StarWars.

    Returns: planets
        planets (list[str]): A list of the homeworlds of sentient speiceis in
            StarWars.
    """
    url = "https://swapi-api.hbtn.io/api/species/?format=json"
    species = []
    planets_url = []
    planets = []

    while url:
        r_json = requests.get(url).json()
        species += r_json['results']
        url = r_json['next']

    for s in species:
        if s['designation'] == 'sentient' and s['homeworld'] is not None:
            planets_url.append(s['homeworld'])

    for planet in planets_url:
        name = requests.get(planet).json()['name']
        planets.append(name)

    return planets


if __name__ == "__main__":
    planets = sentientPlanets()
    for planet in planets:
        print(planet)

    # Expected output
    """
Endor
Naboo
Coruscant
Kamino
Geonosis
Utapau
Kashyyyk
Cato Neimoidia
Rodia
Nal Hutta
unknown
Trandosha
Mon Cala
Sullust
Toydaria
Malastare
Ryloth
Aleen Minor
Vulpter
Troiken
Tund
Cerea
Glee Anselm
Iridonia
Tholoth
Iktotch
Quermia
Dorin
Champala
Mirial
Zolan
Ojom
Skako
Muunilinst
Shili
Kalee
"""
