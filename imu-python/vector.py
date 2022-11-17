#!/usr/bin/env python
#
#   Copyright (C) 2016 Sean D'Epagnier
#
# This Program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

import math


def lmap(*cargs):
    return list(map(*cargs))


def norm(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def normalize(v):
    n = norm(v)
    if n == 0:
        return v
    return lmap(lambda x : x / n, v)


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def sub(a, b):
    return lmap(lambda x, y : x - y, a, b)


def add(a, b):
    return lmap(lambda x, y : x + y, a, b)


def scale(a, m):
    return lmap(lambda x : x*m, a)


def project(a, b):
    return scale(b, dot(a, b)/dot(b, b))


def dist(a, b):
    return norm(sub(a, b))


def dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2
