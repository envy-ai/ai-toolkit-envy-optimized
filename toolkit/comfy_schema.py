"""Read-only checks of API graphs against ComfyUI's /object_info schemas.

This is not Comfy's execution validator: custom VALIDATE_INPUTS, dynamic node
expansion, file contents and model/adapter compatibility still require runtime
verification. No node execution, GPU access or server mutations occur here.
"""
import math


def workflow_schema_errors(workflow, object_info, *, uploaded_images=()):
    """Return actionable static errors without substituting nodes or file choices."""
    errors = []
    if not isinstance(workflow, dict) or not workflow:
        return ['Workflow must be a nonempty API-format node dictionary.']
    if not isinstance(object_info, dict) or not object_info:
        return ['ComfyUI node registry is empty or invalid.']
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            errors.append(f'Node {node_id}: expected an API node dictionary.')
            continue
        name = node.get('class_type')
        schema = object_info.get(name)
        label = f'Node {node_id} ({name})'
        if not isinstance(schema, dict):
            errors.append(f'{label}: node class is not installed on this ComfyUI server.')
            continue
        inputs = node.get('inputs', {})
        if not isinstance(inputs, dict):
            errors.append(f'{label}: inputs must be a dictionary.')
            continue
        sections = schema.get('input', {})
        required = dict(sections.get('required', {}))
        optional = dict(sections.get('optional', {}))
        # V3 Autogrow containers are not themselves API inputs. Comfy expands
        # their templates into dotted child slots; min=0 legitimately accepts
        # no references (e.g. Qwen 2.1 text-to-image). Keep checking each supplied
        # child's declared name/type, and required slots, rather than skipping
        # the whole dynamic group. Mirrors Autogrow._expand_schema_for_dynamic.
        for group, spec in list({**required, **optional}.items()):
            if not isinstance(spec, (list, tuple)) or not spec or spec[0] != 'COMFY_AUTOGROW_V3':
                continue
            try:
                template = spec[1]['template']
                minimum = template['min']
                if 'names' in template:
                    names = template['names']
                else:
                    maximum, prefix = template['max'], template['prefix']
                    if type(maximum) is not int or not 1 <= maximum <= 100 or not isinstance(prefix, str):
                        raise ValueError('invalid prefix template')
                    names = [f'{prefix}{index}' for index in range(maximum)]
                if (not isinstance(names, list) or len(names) > 100 or not all(isinstance(name, str) for name in names)
                        or type(minimum) is not int or not 0 <= minimum <= len(names)):
                    raise ValueError('invalid names/minimum')
                inner = template['input']
                category, children = next((category, children) for category, children in inner.items()
                                           if category in ('required', 'optional') and children)
                child_spec = next(iter(children.values()))
                if not isinstance(child_spec, (list, tuple)) or not child_spec:
                    raise ValueError('invalid child specification')
            except (KeyError, IndexError, TypeError, ValueError, StopIteration):
                errors.append(f'{label}.{group}: installed Autogrow schema is malformed or unsupported.')
                continue
            required.pop(group, None)
            optional.pop(group, None)
            for index, name in enumerate(names):
                target = required if category == 'required' and index < minimum else optional
                target[f'{group}.{name}'] = child_spec
        specs = {**required, **optional}
        hidden = sections.get('hidden', {})
        for key in required:
            if key not in inputs:
                errors.append(f'{label}.{key}: required input is missing.')
        for key, value in inputs.items():
            spec = specs.get(key, hidden.get(key))
            if spec is None:
                errors.append(f'{label}.{key}: input is not declared by the installed node schema.')
                continue
            if not isinstance(spec, (list, tuple)) or not spec:
                # Comfy-injected hidden inputs are advertised as bare type names.
                continue
            kind = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
                source = workflow.get(value[0])
                if not isinstance(source, dict):
                    errors.append(f'{label}.{key}: link refers to missing node {value[0]}.')
                    continue
                outputs = object_info.get(source.get('class_type'), {}).get('output', [])
                if not 0 <= value[1] < len(outputs):
                    errors.append(f'{label}.{key}: node {value[0]} has no output {value[1]}.')
                    continue
                actual = outputs[value[1]]
                # Wildcards, unions and COMBO outputs occur in Easy Use loops.
                # As in Comfy, wildcard links cannot prove a narrower type.
                expected = 'COMBO' if isinstance(kind, list) else kind
                if (isinstance(expected, str) and isinstance(actual, str)
                        and '*' not in (expected, actual)
                        and not set(expected.split(',')).intersection(actual.split(','))):
                    errors.append(f'{label}.{key}: expected {expected}, linked output is {actual}.')
                continue
            choices = kind if isinstance(kind, list) else opts.get('options') if kind == 'COMBO' else None
            if choices is not None:
                # LoadImage advertises only flat input files, but its runtime
                # validator also accepts subfolders. Only trust exact upload
                # receipts from this client, not arbitrary unlisted paths.
                uploaded = (name == 'LoadImage' and key == 'image' and opts.get('image_upload') is True
                            and isinstance(value, str) and value in uploaded_images)
                if value not in choices and not uploaded:
                    errors.append(f'{label}.{key}: {value!r} is not an available choice on this server.')
                continue
            valid = True
            if kind == 'INT':
                valid = type(value) is int
            elif kind == 'FLOAT':
                valid = type(value) in (int, float) and math.isfinite(value)
            elif kind == 'BOOLEAN':
                valid = type(value) is bool
            elif kind == 'STRING':
                valid = isinstance(value, str)
            elif kind != '*':
                errors.append(f'{label}.{key}: {kind} requires a linked node output, not a literal.')
                continue
            if not valid:
                errors.append(f'{label}.{key}: literal must be a finite {kind} value.')
                continue
            if kind in ('INT', 'FLOAT'):
                if 'min' in opts and value < opts['min'] or 'max' in opts and value > opts['max']:
                    errors.append(f'{label}.{key}: {value} is outside the installed node range.')
                # step is a widget increment, not a general validity constraint.
    return errors


def validate_workflow_schema(workflow, object_info, *, uploaded_images=()):
    """Raise one readable error containing all detected static graph problems."""
    errors = workflow_schema_errors(workflow, object_info, uploaded_images=uploaded_images)
    if errors:
        raise ValueError('ComfyUI workflow schema validation failed:\n' + '\n'.join(errors))
