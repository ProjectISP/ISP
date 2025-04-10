import argparse
import os
from obspy import read_inventory

def list_directory(data_path):
    obsfiles = []
    for top_dir, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".xml") or file.endswith(".dataless"):  # Optional filter
                obsfiles.append(os.path.join(top_dir, file))
    obsfiles.sort()
    return obsfiles

def join_meta(data_path: str, output_file: str):
    obsfiles = list_directory(data_path)

    # Initialize an empty inventory
    inv_combined = None

    for meta in obsfiles:
        print(f"Reading: {meta}")
        try:
            inv = read_inventory(meta)
            print(f" → Read {len(inv.networks)} network(s)")
            if inv_combined is None:
                inv_combined = inv
            else:
                inv_combined += inv
        except Exception as e:
            print(f"Failed to read {meta}: {e}")

    if inv_combined is not None:
        print(f"\nWriting combined inventory to: {output_file}")
        inv_combined.write(output_file, format="stationxml")
    else:
        print("No valid metadata found to write.")

def main():
    parser = argparse.ArgumentParser(description="Join multiple ObsPy StationXML metadata files into one.")
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Path to folder containing inventory files (.xml or .dataless)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="combined_station.xml",
        help="Output StationXML filename (default: combined_station.xml)"
    )
    args = parser.parse_args()

    join_meta(args.input, args.output)

if __name__ == "__main__":
    main()


