import copy
import pandas as pd
from gwpy.segments import Segment, SegmentList
from gwsumm.segments import get_segments

_cont_up_times_filename_template = "{ifo}_cont_up_times_from_{start_time}_to_{end_time}.csv"
_cont_down_times_filename_template = "{ifo}_cont_down_times_from_{start_time}_to_{end_time}.csv"

# NOTE These should be the ones to use for O3&O4
_default_flagname_dict = {
    "H1": "H1:DMT-ANALYSIS_READY:1",
    "L1": "L1:DMT-ANALYSIS_READY:1",
    "V1": "V1:ITF_SCIENCE:1",
}

# Any segment shorter than this threshold will be ignored
_default_duration_threshold = 60 # seconds

def convert_seglist_to_df(seglist):
    """
    Convert a SegmentList to a pandas DataFrame

    Parameters
    ----------
    seglist : SegmentList
        The segment list to convert
    
    Returns
    -------
    pd.DataFrame
        The segment list as a DataFrame with columns "start_time", "end_time", and "duration"
    """
    return pd.DataFrame({
        "start_time": [seg.start for seg in seglist],
        "end_time": [seg.end for seg in seglist],
        "duration": [seg.end - seg.start for seg in seglist],
    })

def fetch_segments(
        start_time,
        end_time,
        flagname_dict=_default_flagname_dict,
        duration_threshold=_default_duration_threshold,
        save_to_file=True,
        verbose=True,
    ):
    """
    Fetch the continuous up and down times for the analysis ready flag for each IFO

    Parameters
    ----------
    start_time : int
        The GPS start time
    end_time : int
        The GPS end time
    flagname_dict : dict, optional
        A dictionary with the flag names for each IFO, by default _default_flagname_dict
    duration_threshold : int, optional
        Any segment shorter than this threshold will be ignored, by default _default_duration_threshold
    save_to_file : bool, optional
        Whether to save the up and down times to files, by default True
    verbose : bool, optional
        Whether to print information, by default True

    Returns
    -------
    dict, dict
        Two dictionaries, one with the up times and one with the down times for each IFO
    """
    up_dfs = []
    down_dfs = []

    for ifo in flagname_dict.keys():
        if verbose:
            print(f"Fetching segment lists for {ifo} from {start_time} to {end_time}")

        seglist = get_segments(
            flagname_dict[ifo],
            validity=SegmentList([Segment(start_time, end_time)]),
        )

        up_segment_list = copy.deepcopy(seglist.active)
        indices_to_keep = []

        # Loop over segments where the analysis ready flag is active/on, which are the up times
        for idx in range(len(up_segment_list)):
            _duration = up_segment_list[idx].end - up_segment_list[idx].start
            if _duration > duration_threshold:
                indices_to_keep.append(idx)
    
        up_segment_list = SegmentList([up_segment_list[i] for i in indices_to_keep])
        up_df = convert_seglist_to_df(up_segment_list)

        # Down times are simply the gaps in-between
        down_segment_list = []
        for idx in range(1, len(up_segment_list)):
            _seg = Segment([up_segment_list[idx-1].end, up_segment_list[idx].start])
            down_segment_list.append(_seg)

        down_segment_list = SegmentList(down_segment_list)
        down_df = convert_seglist_to_df(down_segment_list)

        # Save to files
        if save_to_file:
            _up_csv_filename = _cont_up_times_filename_template.format(ifo=ifo, start_time=start_time, end_time=end_time)
            _down_csv_filename = _cont_down_times_filename_template.format(ifo=ifo, start_time=start_time, end_time=end_time)
        
            if verbose:
                print("Saving files to", _up_csv_filename, "and", _down_csv_filename)
            up_df.to_csv(_up_csv_filename, index=False)
            down_df.to_csv(_down_csv_filename, index=False)

        up_dfs.append(up_df)
        down_dfs.append(down_df)

    return dict(zip(flagname_dict.keys(), up_dfs)), dict(zip(flagname_dict.keys(), down_dfs))