# Extension UI modules

A model package registers its python classes in `AI_TOOLKIT_MODELS`
(`extensions/<pkg>/__init__.py`). To show those archs in the web UI (training
form and Generate page) add a `ui.tsx` next to it exporting the UI entries:

```tsx
// extensions/<pkg>/ui.tsx
import type { ModelArch } from '@/app/jobs/new/options';

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  {
    name: 'my_arch', // must equal the python class's `arch`
    label: 'My Model',
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': ['org/my-model', ''],
      'config.process[0].model.quantize': [true, false],
    },
  },
];
```

- The file is read and transpiled by the UI server on every page load
  (`/api/model_archs`), so adding or editing one needs no rebuild. `.ts`,
  `.jsx` and `.js` also work.
- The `ModelArch` shape is in `ui/src/app/jobs/new/options.tsx`; the built-in
  tables in `extensions_built_in/*/ui.tsx` are the reference examples.
- Code runs in the browser. Only these imports are available: `react`,
  `next/link`, `@/helpers/defaultSamples`, `@/components/formInputs`, plus
  `@/app/jobs/new/options`, and type-only imports from `@/types`.
- Entries are merged by `name`; `extensions/` loads after
  `extensions_built_in/`, so an extension can override a built-in entry.
- A module that fails to compile or evaluate is skipped and logged in the
  browser console; the others still load.
- Type check: `cd ui && npm run check_extensions` (uses the repo root
  `tsconfig.json`, which editors also pick up for these files).

Training form model controls use `customModelSelectOptions`. Each option has a
`label`, `getValue(config)`, `onChange(value, config, setJobConfig)`, and optional
`doc`. The existing form renders `type: 'checkbox'` with `Checkbox`,
`type: 'number'` with `NumberInput` (`min` and `step` optional; no blur clamping),
and absent/`'select'` with `SelectInput` and an `options` array. These retain the
standard mouse, Tab, arrow-key, and Space/Enter behavior of the shared controls.

`datasetControlPathListMax` exposes that many ordered reference-folder selectors
using the existing `SelectInput`, stored as `dataset.control_path: string[]`.
It is independent of `datasets.multi_control_paths`, which retains the existing
numbered fields and sample controls. Incomplete slots stay visible while editing;
model validation determines allowed reference counts. Training-only model
packages can set `generate: { disabled: true }` to omit their unsupported native
pipeline from the Generate page while keeping the training selector.

`handleModelArchChange` accepts an optional `{ preserveTrainingChoices: true }`
for component changes that retain an existing training setup (used when entering
or leaving FIBO2). It still applies the destination architecture's controls and
schedule defaults, but retains sample prompts, adapter ranks and memory/cache/
offload choices. It rebuilds model kwargs from the destination defaults, clearing
imported component overrides while retaining destination mode flags. It clears
component paths, pretrained/training/preview adapters, and dataset/sample
references; other sample settings (prompt, seed, strength) stay intact. The caller presents the existing architecture-change
notice. The default call retains previous architecture-switch behavior.
