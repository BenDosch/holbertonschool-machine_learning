#!/usr/bin/env python3
"""Script that takes a GitHub username as an argument and prints their
location."""

import requests
import sys
import time


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user = sys.argv[1]
        r = requests.get(user)

        if r.status_code == 404:
            print('Not found')
        elif r.status_code == 403:
            reset_time = int(r.headers['X-RateLimit-Reset']) - time.time()
            minutes = round(reset_time / 60)
            print('Reset in {} min'.format(minutes))
        elif r.status_code == 200:
            r_json = r.json()
            if r_json['location']:
                print(r_json['location'])
            else:
                print('Not found')
    else:
        print("Please pass a full user url as an argument to the srcipt")
