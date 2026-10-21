import os
from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension import get_all_extensions_process_dict
from toolkit.paths import CONFIG_ROOT

class ExtensionJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device = self.get_conf('device', 'cpu')
        processes = self.get_conf('process')
        if isinstance(processes, list) and processes and all(
            isinstance(process, dict) and process.get('type') == 'APICaptioner'
            for process in processes
        ):
            # Remote captioning must start without discovering GPU extensions.
            # Resolve the existing built-in directly; global discovery imports
            # model packages and initializes every unrelated process class.
            from extensions_built_in.captioner import APICaptionerExtension
            self.process_dict = {APICaptionerExtension.uid: APICaptionerExtension.get_process()}
        else:
            # Retain registration order, overrides and errors for custom,
            # mixed-process and other existing extension jobs.
            self.process_dict = get_all_extensions_process_dict()
        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
