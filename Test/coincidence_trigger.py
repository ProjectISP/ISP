from surfquakecore.coincidence_trigger.coincidence_trigger import CoincidenceTrigger
from surfquakecore.project.surf_project import SurfProject
from surfquakecore.coincidence_trigger.structures import CoincidenceConfig, Kurtosis, STA_LTA, Cluster
from datetime import datetime, timedelta

def parse_datetime(dt_str: str) -> datetime:
    # try with microseconds, fall back if not present
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date string not in expected format: {dt_str}")



if __name__ == '__main__':

    project_file = "/Users/robertocabiecesdiaz/Documents/ISP/isp/examples/Earthquake_location_test/event.pkl"
    span_seconds = 86400
    picking_file = "/Users/robertocabiecesdiaz/Documents/ISP/isp/examples/Earthquake_location_test/output.txt"
    output_folder = "/Users/robertocabiecesdiaz/Documents/ISP/Test/coincidence"
    plot = True

    # --- Decide between time segment or split ---
    sp = SurfProject.load_project(project_file)
    info = sp.get_project_basic_info()
    min_date = info["Start"]
    max_date = info["End"]
    dt1 = parse_datetime(min_date)
    dt2 = parse_datetime(max_date)

    diff = abs(dt2 - dt1)
    if diff < timedelta(days=1):
        sp.get_data_files()
        subprojects = [sp]

    else:
        print(f"[INFO] Splitting into subprojects every {span_seconds} seconds")
        subprojects = sp.split_by_time_spans(
            span_seconds=span_seconds,
            min_date=min_date,
            max_date=max_date,
            file_selection_mode="overlap_threshold",
            verbose=True)

    config = CoincidenceConfig(
        kurtosis_configuration=Kurtosis(CF_decay_win=4.0),
        sta_lta_configuration=STA_LTA(method="classic", sta_win=1.0, lta_win=40.0),
        cluster_configuration=Cluster(
            method_preferred="SNR",
            centroid_radio=60,
            coincidence=3,
            threshold_on=20,
            threshold_off=5,
            fmin=0.5,
            fmax=8.0
        )
    )

    ct = CoincidenceTrigger(subprojects, config, picking_file, output_folder, plot)
    ct.optimized_project_processing()