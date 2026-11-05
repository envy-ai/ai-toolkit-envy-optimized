from typing import Union, OrderedDict

from toolkit.config import get_config


def is_api_caption_job(config):
    if config.get('job') != 'extension':
        return False
    processes = config.get('config', {}).get('process')
    return isinstance(processes, list) and bool(processes) and all(
        isinstance(process, dict) and process.get('type') == 'APICaptioner'
        for process in processes
    )


def get_job(
        config_path: Union[str, dict, OrderedDict],
        name=None,
        *,
        already_parsed=False,
):
    # run.py classifies the parsed config before calling the job factory.
    config = config_path if already_parsed else get_config(config_path, name)
    if not config['job']:
        raise ValueError('config file is invalid. Missing "job" key')

    job = config['job']
    if job == 'extract':
        from jobs import ExtractJob
        return ExtractJob(config)
    if job == 'train':
        from jobs import TrainJob
        return TrainJob(config)
    if job == 'mod':
        from jobs import ModJob
        return ModJob(config)
    if job == 'generate':
        from jobs import GenerateJob
        return GenerateJob(config)
    if job == 'extension':
        from jobs import ExtensionJob
        return ExtensionJob(config)

    # elif job == 'train':
    #     from jobs import TrainJob
    #     return TrainJob(config)
    else:
        raise ValueError(f'Unknown job type {job}')


def run_job(
        config: Union[str, dict, OrderedDict],
        name=None
):
    job = get_job(config, name)
    job.run()
    job.cleanup()
