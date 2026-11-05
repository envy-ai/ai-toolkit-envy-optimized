import os
import sys
from toolkit.dataset_validation import DatasetValidationError
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()
os.environ["HF_XET_HIGH_PERFORMANCE"] = os.getenv("HF_XET_HIGH_PERFORMANCE", "1")
os.environ["HF_HUB_DISABLE_XET"] = os.getenv("HF_HUB_DISABLE_XET", "0")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
seed = None
if "SEED" in os.environ:
    try:
        seed = int(os.environ["SEED"])
    except ValueError:
        print(f"Invalid SEED value: {os.environ['SEED']}. SEED must be an integer.")

sys.path.insert(0, os.getcwd())

# The UI launches jobs with no console; keep anything we shell out to (torch
# compiles, HF git downloads) from flashing a console window. Must come before
# any import that might spawn a subprocess.
from toolkit.win_console import suppress_child_consoles
suppress_child_consoles()

# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

import argparse
from toolkit.config import get_config
from toolkit.job import get_job, is_api_caption_job
from toolkit.print import setup_log_to_file

accelerator = None


def initialize_gpu_runtime():
    """Initialize the existing training runtime once, before model imports."""
    global accelerator
    if accelerator is not None:
        return
    import torch
    if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
        torch.autograd.set_detect_anomaly(True)
    # On unified-memory GPUs, CPU offloading does not free model memory.
    from toolkit.util.unified_memory import apply_unified_memory_patches
    apply_unified_memory_patches()
    if seed is not None:
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    from toolkit.accelerator import get_accelerator
    accelerator = get_accelerator()


def is_main_process():
    if accelerator is None:
        return os.environ.get('RANK', '0') == '0'
    return accelerator.is_main_process


def print_acc(*args, **kwargs):
    if accelerator is None:
        local_main = os.environ.get('LOCAL_RANK', '0') == '0'
    else:
        local_main = accelerator.is_local_main_process
    if local_main:
        print(*args, **kwargs)


def print_end_message(jobs_completed, jobs_failed):
    if not is_main_process():
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if args.log is not None:
        # LOCAL_RANK is supplied by distributed launchers. The API path needs
        # a log before config validation without constructing an Accelerator.
        local_rank = os.environ.get('LOCAL_RANK', '0')
        setup_log_to_file(args.log, is_local_main_process=local_rank == '0')

    announced = False

    def announce_jobs():
        nonlocal announced
        if not announced:
            announced = True
            if is_main_process():
                print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        job = None
        try:
            config = get_config(config_file, args.name)
            if not is_api_caption_job(config):
                initialize_gpu_runtime()
            announce_jobs()
            job = get_job(config, args.name, already_parsed=True)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except DatasetValidationError as e:
            announce_jobs()
            from toolkit.ui_job_status import UIJobStatus
            print_acc(f'\n{e}\n')
            UIJobStatus(e.sqlite_db_path).update_status('stopped', str(e))
            jobs_failed += 1
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                sys.exit(2)
        except Exception as e:
            announce_jobs()
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            if job is not None:
                try:
                    job.process[0].on_error(e)
                except Exception as e2:
                    print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            announce_jobs()
            if job is not None:
                try:
                    job.process[0].on_error(e)
                except Exception as e2:
                    print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_acc("")
                print_acc("========================================")
                print_acc("Job stopped")
                print_acc("========================================")
                sys.exit(0)


if __name__ == '__main__':
    main()
