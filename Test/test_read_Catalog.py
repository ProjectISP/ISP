from obspy import read_events, Catalog
import pandas as pd


class EarthquakeDataExtractor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.latitudes = []
        self.longitudes = []
        self.depths = []
        self.magnitudes = []

    def load_data(self):
        try:
            # Try to read the file as a Catalog object
            catalog = read_events(self.file_path)
            if isinstance(catalog, Catalog):
                print("File contains a valid ObsPy Catalog object. Extracting data...")
                self._extract_from_catalog(catalog)
            else:
                raise ValueError("File does not contain a valid ObsPy Catalog object.")
        except Exception as e:
            print(f"Failed to load as ObsPy Catalog: {e}. Attempting to load with pandas.")
            # If it’s not a Catalog, load the file using pandas
            self._load_with_pandas()

    def _extract_from_catalog(self, catalog):
        # Loop through each event in the catalog
        for event in catalog:
            origin = event.preferred_origin() or event.origins[0]
            latitude = origin.latitude
            longitude = origin.longitude
            depth = origin.depth  # Depth in meters
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            mag_value = magnitude.mag

            # Append data to lists
            self.latitudes.append(latitude)
            self.longitudes.append(longitude)
            self.depths.append(depth / 1000)  # Convert depth to km if desired
            self.magnitudes.append(mag_value)

    def _load_with_pandas(self):
        # Use pandas to load the file
        try:
            data = pd.read_csv(self.file_path)
            print("File loaded successfully with pandas.")
            # Assuming the file contains columns for lat, lon, depth, and magnitude
            self.latitudes = data['latitude'].tolist()
            self.longitudes = data['longitude'].tolist()
            self.depths = data['depth'].tolist()
            self.magnitudes = data['magnitude'].tolist()
        except Exception as e:
            print(f"Error loading file with pandas: {e}")

    def get_data(self):
        return {
            "latitude": self.latitudes,
            "longitude": self.longitudes,
            "depth_km": self.depths,
            "magnitude": self.magnitudes
        }