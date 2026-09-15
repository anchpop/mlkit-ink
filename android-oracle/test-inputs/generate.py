#!/usr/bin/env python3
"""Deterministic hand-drawn polylines, screen coordinates (y grows downward)."""
import json
from pathlib import Path


def write(name, lines, timed):
    strokes = []
    t = 100
    for line in lines:
        points = []
        for a, b in zip(line, line[1:]):
            n = max(1, round(((b[0]-a[0])**2 + (b[1]-a[1])**2)**0.5 / 3))
            points.extend([(round(a[0]+(b[0]-a[0])*i/n, 3),
                            round(a[1]+(b[1]-a[1])*i/n, 3)) for i in range(n)])
        points.append(line[-1])
        stroke = {'x': [p[0] for p in points], 'y': [p[1] for p in points]}
        if timed:
            stroke['t'] = list(range(t, t + 10*len(points), 10))
        t += 10*len(points) + 80
        strokes.append(stroke)
    (Path(__file__).parent / f'{name}.json').write_text(
        json.dumps({'language': 'en-US', 'strokes': strokes}, indent=2) + '\n')


# h: tall downstroke then rounded shoulder; i: short stem then a dot.
write('hi', [
    [(20, 20), (19, 40), (18, 65), (18, 100)],
    [(18, 78), (24, 66), (32, 61), (40, 63), (45, 70), (46, 100)],
    [(66, 63), (65, 82), (65, 100)],
    [(65, 42), (66, 43)],
], True)
# c: open oval; a: closed oval then right stem; t: tall stem then crossbar.
write('cat', [
    [(47, 66), (38, 59), (26, 61), (17, 70), (15, 84), (20, 96), (32, 100), (45, 94)],
    [(85, 67), (76, 60), (65, 63), (57, 74), (57, 87), (65, 98), (76, 98), (85, 87), (85, 66)],
    [(85, 64), (85, 81), (86, 98), (92, 100)],
    [(111, 32), (110, 51), (108, 77), (109, 94), (115, 100), (124, 96)],
    [(98, 62), (113, 60), (126, 60)],
], False)
