"""Inference-only subprocess used by the saved-job sample API."""

import argparse
import json
import os
from pathlib import Path
import traceback

from toolkit.job_samples import checkpoint_step, render_saved_samples, select_checkpoint, validate_render_config


def write_status(filename, value):
    target = Path(filename)
    temporary = target.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    state = {'status': 'running', 'message': 'Validating saved checkpoint…', 'pid': os.getpid()}
    try:
        validate_render_config(request['config'])
        checkpoint = select_checkpoint(Path(request['job_folder']), request['config']['config']['name'])
        state.update(checkpoint=str(checkpoint), step=checkpoint_step(checkpoint))
        if args.validate:
            print(json.dumps(state))
            return
        def update(message):
            state.update(message=message)
            write_status(request['status_path'], state)
        update(f'Rendering saved checkpoint at step {state["step"]}…')
        render_saved_samples(request['config'], request['job_folder'], checkpoint, update)
        state.update(status='completed', message=f'Sample render completed from checkpoint at step {state["step"]}.')
    except BaseException as error:
        state.update(status='error', message=str(error), error=str(error), stack=traceback.format_exc())
        if args.validate:
            raise
        raise
    finally:
        if not args.validate:
            write_status(request['status_path'], state)
            for folder in request['lock_folders']:
                folder = Path(folder)
                owner = json.loads((folder / 'owner.json').read_text())
                if owner['token'] == request['lock_token']:
                    (folder / 'owner.json').unlink()
                    folder.rmdir()


if __name__ == '__main__':
    main()
