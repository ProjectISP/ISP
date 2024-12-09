import os
from surfquakecore.utils.manage_catalog import BuildCatalog, WriteCatalog

path_events_file = '/test_surfquake/outputs/nll/all_loc'
path_source_file = '/test_surfquake/outputs/source/source_summary.txt'
path_mti_summary = "/test_surfquake/outputs/mti/summary_mti.txt"
output_path = "/test_surfquake/outputs/catalog"

format = "QUAKEML"
bc = BuildCatalog(loc_folder=path_events_file, output_path=output_path,
                  source_summary_file=path_source_file, format=format)
bc.build_catalog_loc()

catalog_surf = os.path.join(output_path, "catalog_surf")
catalog_obj = os.path.join(output_path, "catalog_obj.pkl")
wc = WriteCatalog(catalog_obj)
wc.write_catalog_surf(catalog = None, output_path=catalog_surf)