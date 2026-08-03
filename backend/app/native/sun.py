"""Dependency-free sunrise / sunset (the standard NOAA / Wikipedia "Sunrise
equation").

The per-camera "Smart spotlight" (native/spotlight.py) only fires at NIGHT —
local sunset..sunrise for the deploy's configured location. The container ships
no astronomy package (astral / ephem / skyfield), so this module computes the
sun's rise/set with pure ``math`` from the well-known low-precision solar
geometry algorithm. Accuracy is on the order of a minute — vastly better than
needed to gate a night-only spotlight.

Longitude is EAST-positive (western hemisphere negative), matching
``config.longitude``. Times are UNIX epoch seconds (UTC).

Public API:
  * :func:`is_night(now_epoch, lat, lon) -> bool` — the gate the controller
    calls: true when ``now`` is before today's sunrise or at/after today's
    sunset (and always true through a polar night / false through a polar day).
  * :func:`sun_events(now_epoch, lat, lon) -> SunEvents` — the underlying
    sunrise/sunset epochs (+ polar flags), exposed so it is unit-testable.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

# Julian date of the Unix epoch (1970-01-01T00:00:00Z) and seconds per day.
_UNIX_EPOCH_JD = 2440587.5
_SECONDS_PER_DAY = 86400.0
# Julian date of the J2000.0 epoch (2000-01-01T12:00 TT), the algorithm's origin.
_J2000 = 2451545.0
# Fractional-day leap-second correction folded into the mean-solar-time origin.
_LEAP = 0.0009
# Obliquity of the ecliptic (Earth's axial tilt), degrees.
_OBLIQUITY_DEG = 23.44
# Standard sunrise/sunset solar altitude: the disc's upper limb touches the
# horizon (incl. atmospheric refraction) when the center is 0.833° below it.
_SUNRISE_ALTITUDE_DEG = -0.833


class SunEvents(NamedTuple):
    """Sunrise / sunset for the solar day nearest ``now`` (UTC epoch seconds).

    ``sunrise`` / ``sunset`` are None in the polar cases, disambiguated by the
    flags: ``always_up`` (midnight sun — the sun never sets that day) and
    ``always_down`` (polar night — the sun never rises)."""

    sunrise: Optional[float]
    sunset: Optional[float]
    always_up: bool
    always_down: bool


def _to_jd(epoch: float) -> float:
    return epoch / _SECONDS_PER_DAY + _UNIX_EPOCH_JD


def _to_epoch(jd: float) -> float:
    return (jd - _UNIX_EPOCH_JD) * _SECONDS_PER_DAY


def sun_events(now_epoch: float, lat: float, lon: float) -> SunEvents:
    """Compute sunrise/sunset for the solar day whose local noon is nearest to
    ``now_epoch`` at (``lat``, ``lon``). Pure computation; never raises."""
    lat_r = math.radians(lat)
    # The algorithm is written in west-positive longitude; config longitude is
    # east-positive, so negate.
    lon_west = -lon

    jd = _to_jd(now_epoch)
    # Julian day count since J2000 for the day whose mean solar noon is nearest
    # ``now`` (round -> the closest solar day, so a time just past midnight maps
    # to the correct calendar day's rise/set).
    n = round(jd - _J2000 - _LEAP - lon_west / 360.0)
    # Approximate Julian date of that day's mean solar noon.
    j_star = _J2000 + _LEAP + lon_west / 360.0 + n

    # Solar mean anomaly (deg).
    M = (357.5291 + 0.98560028 * (j_star - _J2000)) % 360.0
    M_r = math.radians(M)
    # Equation of the center (deg).
    C = (1.9148 * math.sin(M_r)
         + 0.0200 * math.sin(2 * M_r)
         + 0.0003 * math.sin(3 * M_r))
    # Ecliptic longitude of the sun (deg).
    lam = (M + C + 180.0 + 102.9372) % 360.0
    lam_r = math.radians(lam)
    # Julian date of solar transit (true solar noon).
    j_transit = (j_star
                 + 0.0053 * math.sin(M_r)
                 - 0.0069 * math.sin(2 * lam_r))
    # Declination of the sun.
    sin_decl = math.sin(lam_r) * math.sin(math.radians(_OBLIQUITY_DEG))
    decl = math.asin(sin_decl)

    # Hour angle of sunrise/sunset. |cos| > 1 means the sun never reaches the
    # sunrise altitude that day: > 1 -> it stays below (polar night), < -1 ->
    # it stays above (midnight sun).
    cos_omega = (
        (math.sin(math.radians(_SUNRISE_ALTITUDE_DEG)) - math.sin(lat_r) * sin_decl)
        / (math.cos(lat_r) * math.cos(decl))
    )
    if cos_omega >= 1.0:
        return SunEvents(None, None, always_up=False, always_down=True)
    if cos_omega <= -1.0:
        return SunEvents(None, None, always_up=True, always_down=False)

    omega_deg = math.degrees(math.acos(cos_omega))
    j_rise = j_transit - omega_deg / 360.0
    j_set = j_transit + omega_deg / 360.0
    return SunEvents(_to_epoch(j_rise), _to_epoch(j_set), always_up=False, always_down=False)


def is_night(now_epoch: float, lat: float, lon: float) -> bool:
    """True when it is night (local sunset..sunrise) at (``lat``, ``lon``) for
    the instant ``now_epoch``. Handles the polar edge cases and never raises."""
    ev = sun_events(now_epoch, lat, lon)
    if ev.always_down:
        return True
    if ev.always_up:
        return False
    assert ev.sunrise is not None and ev.sunset is not None
    return now_epoch < ev.sunrise or now_epoch >= ev.sunset
