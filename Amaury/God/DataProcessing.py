import datetime
import logging
import time
from typing import List, Dict

import numpy as np
import scipy.optimize

import God.SaveAndLoad as SaveAndLoad
from God.Physics import Physics
from God.Sky import Sky

log = logging.getLogger('DataProcessing')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


def get_group_size_occurences(groups: List[list]) -> List[int]:
    size_groups = [len(group) for group in groups if len(group) > 0]
    max_size_group = np.max(size_groups)
    size_x = range(1, max_size_group + 1)
    return [size_groups.count(i) for i in size_x]


class Processor:
    simulation_params: dict
    frames: list
    data_golders: Dict[str, list]
    options: dict
    to_process: dict

    def __init__(self):
        self.simulation_params = None
        self.frames = None
        self.to_process = dict()
        self.data_holders = dict()
        self.options = None

    def load_data(self, input_file: str) -> None:
        # load data
        data = SaveAndLoad.load_data(input_file)
        self.frames = data["frames"]
        self.simulation_params = data["parameters"]

        log.info("Processor: Got %d frames" % len(self.frames))

    def chose_what_to_process(self, to_process: List[str]) -> None:
        self.to_process["groups"] = "groups" in to_process
        self.to_process["group_size"] = "group_size" in to_process
        self.to_process["group_size_avg"] = "group_size_avg" in to_process
        self.to_process["group_size_avg_fit"] = self.to_process["group_size_avg"] and (
                "group_size_avg_fit" in to_process)
        self.to_process["avg_speed"] = "avg_speed" in to_process
        self.to_process["avg_angle"] = "avg_angle" in to_process
        self.to_process["correlations"] = "correlations" in to_process
        self.to_process["correlations_fit"] = self.to_process["correlations"] and ("correlations_fit" in to_process)

        keys_to_process = [k for k, v in self.to_process.items() if v]
        log.info("The following properties will be processed: %s" % keys_to_process)

    def init_data_holders(self) -> None:
        if self.to_process["avg_speed"]:
            self.data_holders["avg_speed"] = []
        if self.to_process["avg_angle"]:
            self.data_holders["avg_angle"] = []
        if self.to_process["groups"]:
            self.data_holders["groups"] = []
        if self.to_process["group_size"]:
            self.data_holders["group_size"] = []
        if self.to_process["group_size_avg"]:
            self.data_holders["group_size_avg"] = []
            self.data_holders["group_size_combined"] = []
        if self.to_process["group_size_avg_fit"]:
            self.data_holders["group_size_avg_fit"] = []
        if self.to_process["correlations"]:
            self.data_holders["correlations"] = []
        if self.to_process["correlations_fit"]:
            self.data_holders["correlations_fit"] = []

    def process(self, input_file: str, output_folder: str, to_process: list, verbose_prop: float = .1,
                options: dict = None) -> None:
        self.options = {} if options is None else options
        if "correlations_stochastic_points" not in self.options:
            self.options["correlations_stochastic_points"] = 2000
        if "fit_spatial_points" not in self.options:
            self.options["fit_spatial_points"] = 100

        # load data
        self.load_data(input_file)

        # chose what to process
        self.chose_what_to_process(to_process)

        # init data holders
        self.init_data_holders()

        # process each frame
        frame_number = -1
        total_frames = len(self.frames)
        start_t = time.time()
        log.info("Processing start at t=%s" % datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'))
        for frame in self.frames:
            frame_number += 1
            if frame_number % (1 + int(total_frames * verbose_prop)) == 0:
                time_per_frame = (time.time() - start_t) / (frame_number + 1)
                remaining_time = time_per_frame * (total_frames - frame_number)
                log.info("Processing frame %d/%d - remaining est. %dh %dm %ds" % (frame_number, total_frames,
                                                                                  remaining_time // 3600 % 24,
                                                                                  remaining_time // 60 % 60,
                                                                                  remaining_time % 60,))

            self.process_frame(frame, frame_number)

        elapsed = time.time() - start_t
        log.info(
            "Processing ended at t=%s, elapsed: %dh %dm %ds" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                elapsed // 3600 % 24, elapsed // 60 % 60, elapsed % 60))
        # save results
        self.save_results(output_folder)
        log.info("Processing results saved")

    def process_avg_speed(self, sky: Sky) -> None:
        self.data_holders["avg_speed"].append(sky.get_avg_speed())

    def process_avg_angle(self, sky: Sky) -> None:
        self.data_holders["avg_angle"].append(sky.get_avg_angle())

    def process_groups(self, sky: Sky, birds_to_group: dict):
        self.data_holders["groups"].append([birds_to_group[bird] for bird in sky.birds])

    def process_group_size(self, size_occurences: List[int]) -> None:
        self.data_holders["group_size"].append(size_occurences)

    def process_group_size_avg(self, size_occurences: List[int], frame_number: int) -> None:
        group_size_combined = self.data_holders["group_size_combined"]

        # adjust size of group_size_combined if "new sizes" have appeared
        size_diff = len(size_occurences) - len(group_size_combined)
        if size_diff > 0:
            group_size_combined += [0] * size_diff
        # register sizes observes this frame
        for i in range(len(size_occurences)):
            group_size_combined[i] += size_occurences[i]

        self.data_holders["group_size_avg"].append(np.array(group_size_combined) / (frame_number + 1))

    def process_group_size_avg_fit(self) -> None:
        def fit(x, a1, b1):
            return b1 * x ** a1

        group_size_avg = self.data_holders["group_size_avg"][-1]
        size_x = np.array(range(1, len(group_size_avg) + 1))

        try:
            popt, _ = scipy.optimize.curve_fit(fit, size_x, group_size_avg)
            a, b = popt

            residuals = group_size_avg - fit(size_x, a, b)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((group_size_avg - np.mean(group_size_avg)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            self.data_holders["group_size_avg_fit"].append([a, b, r_squared])
        except Exception as e:
            log.warning("Exception in group size fit: %s" % e)
            self.data_holders["group_size_avg_fit"].append(None)

    def process_correlations(self, sky: Sky, L: float) -> None:
        correlations_stochastic_points = self.options["correlations_stochastic_points"]
        dists, corrs = sky.get_angles_correlations(n=correlations_stochastic_points)

        # average over spatial fixed intervals
        space_points = self.options["fit_spatial_points"]
        regular_dists = np.linspace(0, L, space_points)
        regular_corrs = []
        for i in range(len(regular_dists) - 1):
            mask = np.logical_and(dists > regular_dists[i], dists < regular_dists[i + 1])
            if len(corrs[mask]) == 0:  # no points in interval
                mean = 0
            else:
                mean = np.mean(corrs[mask])
            regular_corrs.append(mean)
        regular_corrs = np.array(regular_corrs)

        self.data_holders["correlations"].append(regular_corrs)

    def process_correlations_fit(self, L: float) -> None:
        def func_fit(x, a1, zeta1):
            return a1 * np.exp(- x / zeta1)

        correlations = self.data_holders["correlations"][-1]
        space_points = self.options.get("fit_spatial_points", 100)
        regular_dists, step = np.linspace(0, L, space_points, retstep=True, endpoint=False)
        regular_dists = regular_dists[:-1] + step / 2  # Offset to account for averaging intervals
        try:
            popt, _ = scipy.optimize.curve_fit(func_fit, regular_dists, correlations)
            a, zeta = popt

            residuals = correlations - func_fit(regular_dists, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((correlations - np.mean(correlations)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            self.data_holders["correlations_fit"].append([a, zeta, r_squared])
        except Exception as e:
            log.warning("Exception in correlation fit: %s" % e)
            self.data_holders["correlations_fit"].append(None)

    def process_frame(self, frame: list, frame_number: int):
        # get simulation parameters
        L = self.simulation_params["L"]
        eta = self.simulation_params["eta"]
        interaction_radius = self.simulation_params["interaction_radius"]

        # recreate environment
        if not (self.to_process[
            "group_size"]):  # no need to process interactions > big grid for faster perfs and less memory usage
            interaction_radius = 2 * L
        sky = SaveAndLoad.recreate_frame(frame, L, interaction_radius / 2)
        physics = Physics(sky, interaction_radius, eta)

        if self.to_process["avg_speed"]:
            self.process_avg_speed(sky)

        if self.to_process["avg_angle"]:
            self.process_avg_angle(sky)

        if self.to_process["groups"] or self.to_process["group_size"] or self.to_process["group_size_avg"]:
            groups, bird_to_group = physics.get_groups()
            size_occurences = get_group_size_occurences(groups)
            if self.to_process["groups"]:
                self.process_groups(sky, bird_to_group)
            if self.to_process["group_size"]:
                self.process_group_size(size_occurences)
            if self.to_process["group_size_avg"]:
                self.process_group_size_avg(size_occurences, frame_number)
                if self.to_process["group_size_avg_fit"]:
                    self.process_group_size_avg_fit()

        if self.to_process["correlations"]:
            self.process_correlations(sky, L)
            if self.to_process["correlations_fit"]:
                self.process_correlations_fit(L)

    def save_results(self, output_file: str):
        def save_prop_name(prop_name: str):
            SaveAndLoad.save_data_dirname(self.data_holders[prop_name], output_file, "%s.json" % prop_name)

        # save the actual data
        simple_propreties = ["avg_speed", "avg_angle", "groups", "group_size", "group_size_avg", "group_size_avg_fit",
                             "correlations", "correlations_fit"]
        for property_name in simple_propreties:
            if self.to_process[property_name]:
                save_prop_name(property_name)

        # save the simulation's parameters
        self.simulation_params["processing_options"] = self.options
        SaveAndLoad.save_data_dirname(self.simulation_params, output_file, "processing_parameters.json")
