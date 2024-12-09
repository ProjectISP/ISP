from surfquakecore.real.real_core import RealCore

# Inventory Information
inventory_path = "/Users/robertocabiecesdiaz/Documents/test_surfquake/inputs/metadata/inv_all.xml"

# picking Output of PhaseNet
picks_path = '/Users/robertocabiecesdiaz/Documents/test_surfquake/my_test/picks'

# Set working_directory and output
working_directory = '/Users/robertocabiecesdiaz/Documents/test_surfquake/my_test/real'
output_directory = '/Users/robertocabiecesdiaz/Documents/test_surfquake/my_test/real/output'

# Set path to REAL configuration
config_path = '/Users/robertocabiecesdiaz/Documents/test_surfquake/inputs/configs/real_config.ini'
# Run association
rc = RealCore(inventory_path, config_path, picks_path, working_directory, output_directory)
rc.run_real()
print("End of Events AssociationProcess, please see for results: ", output_directory)