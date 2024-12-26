#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
geodetic_conversion
"""

import math


def calculate_destination_coordinates(origin_lat, origin_lon, distance_east_west_km, distance_north_south_km):
    # Earth radius in kilometers
    earth_radius_km = 6371.0

    # Convert distances to radians
    delta_lat = distance_north_south_km / earth_radius_km
    delta_lon = distance_east_west_km / (earth_radius_km * math.cos(math.radians(origin_lat)))

    # Calculate destination coordinates
    dest_lat = origin_lat + math.degrees(delta_lat)
    dest_lon = origin_lon + math.degrees(delta_lon)

    return dest_lat, dest_lon


# def calculate_destination_coordinates_vincenty(origin_latitude, origin_longitude, distance_east_west_km, distance_north_south_km):
#     # WGS84 ellipsoid parameters
#     semi_major_axis = 6378137.0  # semi-major axis in meters
#     inverse_flattening = 298.257223563  # inverse flattening
#
#     # Calculate semi-minor axis
#     semi_minor_axis = semi_major_axis * (1 - 1 / inverse_flattening)
#
#     # Convert distances to meters
#     delta_north_south = distance_north_south_km * 1000
#     delta_east_west = distance_east_west_km * 1000
#
#     # Initial values
#     phi1 = math.radians(origin_latitude)
#     lambda1 = math.radians(origin_longitude)
#
#     U1 = math.atan((1 - 1 / inverse_flattening) * math.tan(phi1))
#     sigma1 = math.atan2(math.tan(U1), math.cos(lambda1))
#     sin_alpha = math.cos(U1) * math.sin(lambda1)
#     cos2_alpha = 1 - sin_alpha**2
#
#     u2 = cos2_alpha * (semi_major_axis**2 - semi_minor_axis**2) / semi_minor_axis**2
#     A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
#     B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
#
#     sigma = delta_sigma = delta_sigma_prev = delta_lambda = 1
#     iter_limit = 1000
#
#     while delta_sigma > 1e-12 and iter_limit > 0:
#         cos2_sigma_m = math.cos(2 * sigma1 + sigma)
#         sin_sigma = math.sin(sigma)
#         cos_sigma = math.cos(sigma)
#         delta_sigma = B * sin_sigma * (cos2_sigma_m + B / 4 * (
#             cos_sigma * (-1 + 2 * cos2_sigma_m**2) - B / 6 * cos2_sigma_m * (
#                 -3 + 4 * sin_sigma**2) * (-3 + 4 * cos2_sigma_m**2)))
#
#         sigma_prime = sigma
#         sigma = delta_sigma + sigma1
#
#         sin_alpha = math.cos(U1) * math.sin(lambda1)
#         cos2_alpha = 1 - sin_alpha**2
#
#         cos2_sigma_m = math.cos(2 * sigma1 + sigma)
#         cos_sigma = math.cos(sigma)
#         delta_lambda = B * sin_sigma * (cos2_sigma_m + B / 4 * (
#             cos_sigma * (-1 + 2 * cos2_sigma_m**2) - B / 6 * cos2_sigma_m * (
#                 -3 + 4 * sin_sigma**2) * (-3 + 4 * cos2_sigma_m**2)))
#
#         lambda_diff = delta_lambda
#
#         iter_limit -= 1
#
#     if iter_limit == 0:
#         raise ValueError("Vincenty formulae did not converge")
#
#     u2 = cos2_alpha * (semi_major_axis**2 - semi_minor_axis**2) / semi_minor_axis**2
#     A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
#     B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
#     delta_sigma_prime = B * sin_sigma * (cos2_sigma_m + 1 / 4 * B * (
#         cos_sigma * (-1 + 2 * cos2_sigma_m**2) - 1 / 6 * B * cos2_sigma_m * (
#             -3 + 4 * sin_sigma**2) * (-3 + 4 * cos2_sigma_m**2)))
#
#     phi2 = math.atan2(
#         math.sin(U1) * math.cos(sigma) + math.cos(U1) * math.sin(sigma) * math.cos(lambda1),
#         (1 - 1 / inverse_flattening) * math.sqrt(sin_alpha**2 + (math.sin(U1) * math.sin(sigma) - math.cos(U1) * math.cos(sigma) * math.cos(lambda1))**2)
#     )
#
#     lambda2 = lambda1 + math.atan2(
#         math.sin(sigma) * math.sin(lambda1),
#         math.cos(U1) * math.cos(sigma) - math.sin(U1) * math.sin(sigma) * math.cos(lambda1)
#     )
#
#     destination_latitude = math.degrees(phi2)
#     destination_longitude = math.degrees(lambda2)
#
#     return destination_latitude, destination_longitude

if __name__ == "__main__":
    # Example usage
    origin_latitude = 40.7128  # Example origin latitude
    origin_longitude = -74.0060  # Example origin longitude
    distance_east_west = -100  # Example distance towards East or West in kilometers
    distance_north_south = -50  # Example distance towards North or South in kilometers

    destination_latitude, destination_longitude = calculate_destination_coordinates(
        origin_latitude, origin_longitude, distance_east_west, distance_north_south)

    print(f"Destination Latitude: {destination_latitude}, Destination Longitude: {destination_longitude}")