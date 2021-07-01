#!/usr/bin/env python

from astropy import units
from astropy.coordinates import Angle
import json
import math
import redis
import scipy

REDIS = redis.Redis(host="localhost", port=6379, db=0)

FREQUENCY = float(REDIS.get("array_1:current_obs:frequency"))

TARGET_LIST = json.loads(REDIS.get("array_1:current_obs:target_list"))

RADIUS = 0.5 * ((2.998e8 / float(FREQUENCY)) / 1000) * 180 / math.pi


class Point(object):
    def __init__(self, source_id, ra, decl, priority):
        self.source_id = source_id
        self.ra = ra
        self.decl = decl
        self.priority = priority


POINTS = [
    Point(*args)
    for args in zip(
        TARGET_LIST["source_id"],
        TARGET_LIST["ra"],
        TARGET_LIST["decl"],
        TARGET_LIST["priority"],
    )
]


def main():
    pass


if __name__ == "__main__":
    main()
