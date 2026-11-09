#!/usr/bin/env python3
"""
Builds the grids of MV grid districts from an egon-data database and saves
them as CSV files.

The districts are worked off one after another from a queue by a fixed number
of parallel processes, each district in its own process. A district only
starts when enough RAM is free; if the free RAM falls below a critical level,
the most recently started district is stopped and queued again.

Each finished district is recorded in status.csv in the output directory
(status, duration, peak RAM, error). Districts recorded as ok are skipped
when the run is started again with the same output directory, so an
interrupted run continues where it stopped. The log of each district is in
logs/<district>.log.

Examples
--------
    run_egon_data_districts.py 36220 36326
    run_egon_data_districts.py --all --processes 10 \
        --output /storage/JoDa/ding0/results/status2024
"""
import argparse
import csv
import logging
import os
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime

import psutil

logger = logging.getLogger("run_egon_data_districts")

GB = 1024 ** 3
STATUS_COLUMNS = ["district", "status", "seconds", "peak_ram_gb", "error"]
# a district stopped for lack of RAM is queued again this often
MAX_RAM_RETRIES = 2


# Building one district (runs in its own process) ##############################

def connect(args):
    from sqlalchemy.orm import sessionmaker
    from ding0.tools import database

    engine = database.get_engine(egon_data_config=args.egon_data_config)
    return sessionmaker(bind=engine)()


def network(session, args):
    from ding0.core import NetworkDing0
    from ding0.tools import planning_profiles

    return NetworkDing0(
        name="network", session=session, scenario=args.scenario,
        planning_profile=args.planning_profile or planning_profiles.DEFAULT_PROFILE)


def build_district(district_id, args):
    """Builds and saves the grid of one district, returns the error or None."""
    from ding0.tools.egon_data_integration import get_mv_data

    session = connect(args)
    nd = network(session, args)
    if get_mv_data(nd.orm, session, [district_id]).empty:
        return "MV grid district not in the database"
    messages = nd.run_ding0(session=session, mv_grid_districts_no=[district_id])
    # the validation removes invalid districts; the messages also hold
    # warnings, e.g. of the LV clustering, of districts that are built
    if not list(nd.mv_grid_districts()):
        return f"invalid MV grid district: {messages}"
    nd.to_csv(args.output)
    return None


def error_file(output, district_id):
    return os.path.join(output, "logs", f"{district_id}.error")


def worker(district_id, args):
    """Entry point of a district process; the error goes to an error file."""
    try:
        error = build_district(district_id, args)
    except Exception:
        error = traceback.format_exc()
    if error:
        logger.error(f"district {district_id} failed:\n{error}")
        with open(error_file(args.output, district_id), "w") as f:
            f.write(error)
        sys.exit(1)


def all_districts(args):
    """IDs of all MV grid districts with an HV/MV station in the database."""
    session = connect(args)
    orm = network(session, args).orm
    districts, stations = orm["orm_mv_grid_districts"], orm["orm_mv_stations"]
    query = (
        session.query(districts.bus_id)
        .join(stations, districts.bus_id == stations.bus_id)
        .filter(orm["version_condition_mvgd"])
        .filter(orm["version_condition_mv_stations"])
        .order_by(districts.bus_id)
    )
    return [int(row[0]) for row in query]


# Scheduling the districts ####################################################

def read_done(status_file):
    if not os.path.isfile(status_file):
        return set()
    with open(status_file) as f:
        return {int(row["district"]) for row in csv.DictReader(f)
                if row["status"] == "ok"}


def write_status(status_file, row):
    new = not os.path.isfile(status_file)
    with open(status_file, "a", newline="") as f:
        writer = csv.DictWriter(f, STATUS_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def ram_of(process):
    """RAM (RSS) of a process and its children in bytes."""
    try:
        processes = [process] + process.children(recursive=True)
        return sum(p.memory_info().rss for p in processes)
    except psutil.NoSuchProcess:
        return 0


class District:
    """A running district process."""

    def __init__(self, district_id, args, worker_args):
        self.id = district_id
        self.start = time.time()
        self.peak_ram = 0
        self.error_file = error_file(args.output, district_id)
        if os.path.isfile(self.error_file):
            os.remove(self.error_file)
        self.log = open(os.path.join(args.output, "logs", f"{district_id}.log"), "w")
        # one thread per process, so that a process uses one core
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        self.popen = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", str(district_id)]
            + worker_args,
            stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.process = psutil.Process(self.popen.pid)

    def sample_ram(self):
        self.peak_ram = max(self.peak_ram, ram_of(self.process))
        return self.peak_ram

    def stop(self):
        for p in self.process.children(recursive=True) + [self.process]:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        self.popen.wait()
        self.log.close()

    def error(self):
        if os.path.isfile(self.error_file):
            with open(self.error_file) as f:
                return f.read()
        return f"process ended with code {self.popen.returncode}"

    def result(self, status, error=""):
        self.log.close()
        return {"district": self.id, "status": status,
                "seconds": round(time.time() - self.start),
                "peak_ram_gb": round(self.peak_ram / GB, 1),
                "error": error.strip().splitlines()[-1] if error.strip() else ""}


def schedule(districts, args, worker_args):
    """Works off the districts with args.processes parallel processes."""
    status_file = os.path.join(args.output, "status.csv")
    queue = deque(districts)
    ram_retries = {}
    running = []
    counts = {"ok": 0, "failed": 0}
    last_report = 0

    while queue or running:
        available = psutil.virtual_memory().available

        # finished districts
        for district in [d for d in running if d.popen.poll() is not None]:
            running.remove(district)
            status = "ok" if district.popen.returncode == 0 else "failed"
            error = district.error() if status == "failed" else ""
            row = district.result(status, error)
            write_status(status_file, row)
            counts[status] += 1
            logger.info(f"district {district.id} {status} in {row['seconds']} s, "
                        f"peak RAM {row['peak_ram_gb']} GB"
                        + (f": {row['error']}" if row["error"] else ""))

        for district in running:
            district.sample_ram()

        # too little RAM: stop the most recently started district
        if available < args.critical_free_ram * GB and len(running) > 1:
            district = max(running, key=lambda d: d.start)
            running.remove(district)
            district.stop()
            ram_retries[district.id] = ram_retries.get(district.id, 0) + 1
            if ram_retries[district.id] > MAX_RAM_RETRIES:
                write_status(status_file, district.result(
                    "failed", "stopped repeatedly for lack of RAM"))
                counts["failed"] += 1
            else:
                queue.append(district.id)
            logger.warning(f"only {available / GB:.0f} GB RAM free, district "
                           f"{district.id} stopped and queued again")
            time.sleep(10)
            continue

        # start districts while processes and RAM are free
        if (queue and len(running) < args.processes
                and available >= args.min_free_ram * GB):
            district = District(queue.popleft(), args, worker_args)
            running.append(district)
            # give the district time to allocate memory before the next starts
            time.sleep(args.start_interval)
            continue

        if time.time() - last_report > 300:
            last_report = time.time()
            logger.info(
                f"{counts['ok']} ok, {counts['failed']} failed, "
                f"{len(running)} running, {len(queue)} queued, "
                f"{available / GB:.0f} GB RAM free, districts running: "
                + ", ".join(f"{d.id} ({d.peak_ram / GB:.1f} GB)" for d in running))
        time.sleep(2)

    return counts


# Command line ################################################################

def setup_logging(output):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(output, "run.log"))])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("districts", nargs="*", type=int,
                        help="IDs (bus_id) of the MV grid districts")
    parser.add_argument("--all", action="store_true",
                        help="all MV grid districts of the database")
    parser.add_argument("--districts-file",
                        help="file with one district ID per line")
    parser.add_argument("--output", default=None,
                        help="output directory (default ~/ding0_results/<timestamp>)")
    parser.add_argument("--processes", type=int, default=10,
                        help="districts built in parallel")
    parser.add_argument("--min-free-ram", type=float, default=40,
                        help="GB of free RAM needed to start a district")
    parser.add_argument("--critical-free-ram", type=float, default=15,
                        help="GB of free RAM below which a district is stopped")
    parser.add_argument("--start-interval", type=float, default=20,
                        help="seconds between starting two districts")
    parser.add_argument("--scenario", default=None,
                        help="egon-data scenario (default from config_db_tables.cfg)")
    parser.add_argument("--planning-profile", default=None,
                        help="planning profile (default: ding0's default profile)")
    parser.add_argument("--egon-data-config", default=None,
                        help="egon-data YAML (default from config_db_credentials.cfg)")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            "~", "ding0_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    args.output = os.path.abspath(os.path.expanduser(args.output))

    if args.worker is not None:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        for name in ["paramiko", "sshtunnel", "saio", "pypsa"]:
            logging.getLogger(name).setLevel(logging.WARNING)
        worker(args.worker, args)
        return

    os.makedirs(os.path.join(args.output, "logs"), exist_ok=True)
    setup_logging(args.output)

    districts = list(args.districts)
    if args.districts_file:
        with open(os.path.expanduser(args.districts_file)) as f:
            districts += [int(line) for line in f if line.strip()]
    if args.all:
        districts += all_districts(args)
    if not districts:
        parser.error("no districts given")
    done = read_done(os.path.join(args.output, "status.csv"))
    todo = [d for d in dict.fromkeys(districts) if d not in done]

    # passed on to the district processes
    worker_args = ["--output", args.output]
    for option in ["scenario", "planning_profile", "egon_data_config"]:
        if getattr(args, option):
            worker_args += ["--" + option.replace("_", "-"), getattr(args, option)]

    logger.info(f"{len(todo)} districts to build ({len(done)} already done), "
                f"{args.processes} processes, output {args.output}")
    start = time.time()
    counts = schedule(todo, args, worker_args)
    logger.info(f"done in {(time.time() - start) / 3600:.1f} h: "
                f"{counts['ok']} ok, {counts['failed']} failed")


if __name__ == "__main__":
    main()
