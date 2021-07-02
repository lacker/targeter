#!/usr/bin/env python

import json
import math
import numpy as np
import redis
from scipy.spatial import KDTree

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


# Parse the json into Point objects for convenience
POINTS = [
    Point(*args)
    for args in zip(
        TARGET_LIST["source_id"],
        TARGET_LIST["ra"],
        TARGET_LIST["decl"],
        TARGET_LIST["priority"],
    )
]


def intersect_two_circles(x0, y0, r0, x1, y1, r1):
    """
    Finding the intersections of two circles.

    Thanks to:
    https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
    """
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    if d > r0 + r1:
        raise ValueError("non-intersecting")

    if d < abs(r0 - r1):
        raise ValueError("one circle within the other")

    if d == 0 and r0 == r1:
        raise ValueError("coincident circles")

    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r0 ** 2 - a ** 2)
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d
    x3 = x2 + h * (y1 - y0) / d
    y3 = y2 - h * (x1 - x0) / d

    x4 = x2 - h * (y1 - y0) / d
    y4 = y2 + h * (x1 - x0) / d

    return ((x3, y3), (x4, y4))


def main():
    arr = np.array([[p.ra, p.decl] for p in POINTS])
    tree = KDTree(arr)

    # Find all pairs of points that could be captured by a single observation
    pairs = tree.query_pairs(2 * RADIUS)
    print("there are", len(pairs), "pairs")

    # A list of (ra, decl) coordinates for the center of possible circles
    candidate_centers = []

    # Add one center for each of the points that aren't part of any pairs
    in_a_pair = set()
    for i, j in pairs:
        in_a_pair.add(i)
        in_a_pair.add(j)
    for i in range(len(POINTS)):
        if i not in in_a_pair:
            p = POINTS[i]
            candidate_centers.append((p.ra, p.decl))

    # Add a center for each pair of points that are close to each other
    for i0, i1 in pairs:
        p0 = POINTS[i0]
        p1 = POINTS[i1]
        # For each pair, find two points that are a bit less than RADIUS away from each point.
        # These are the possible centers of the circle.
        # TODO: make the mathematical argument of this algorithm's sufficiency clearer
        r = 0.9999 * RADIUS
        try:
            c0, c1 = intersect_two_circles(p0.ra, p0.decl, r, p1.ra, p1.decl, r)
            candidate_centers.append(c0)
            candidate_centers.append(c1)
        except ValueError:
            continue

    # For each circle-center, find the member points in it
    print("there are", len(candidate_centers), "candidates for circle-centers")
    candidate_members = tree.query_ball_point(
        candidate_centers, RADIUS, return_sorted=True
    )

    # Filter out any candidates whose members are the same as a previous candidate
    centers = []
    members = []
    seen = set()
    for c, member_list in zip(candidate_centers, candidate_members):
        key = tuple(member_list)
        if key in seen:
            continue
        seen.add(key)
        centers.append(c)
        members.append(member_list)

    print(
        "after removing functional duplicates, there are",
        len(centers),
        "possible circles",
    )


if __name__ == "__main__":
    main()
