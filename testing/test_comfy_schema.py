import copy
import unittest
from unittest.mock import patch

from toolkit.comfy_schema import validate_workflow_schema, workflow_schema_errors


class ComfySchemaTests(unittest.TestCase):
    def setUp(self):
        self.registry = {
            'Source': {'input': {'required': {'name': [['model.safetensors']]}}, 'output': ['MODEL', 'INT', '*']},
            'Target': {'input': {'required': {'model': ['MODEL'], 'steps': ['INT', {'min': 1, 'max': 100}],
                'cfg': ['FLOAT', {'min': 0}], 'enabled': ['BOOLEAN'], 'text': ['STRING'],
                'sampler': ['COMBO', {'options': ['euler']}]}}, 'output': []},
        }
        self.graph = {
            '1': {'class_type': 'Source', 'inputs': {'name': 'model.safetensors'}},
            '2': {'class_type': 'Target', 'inputs': {'model': ['1', 0], 'steps': 20, 'cfg': 3,
                'enabled': True, 'text': '', 'sampler': 'euler'}},
        }

    def test_valid_graph_is_read_only(self):
        original = copy.deepcopy((self.graph, self.registry))
        validate_workflow_schema(self.graph, self.registry)
        self.assertEqual((self.graph, self.registry), original)

    def test_missing_class_and_required_inputs(self):
        self.graph['1']['class_type'] = 'Missing'
        del self.graph['2']['inputs']['text']
        with self.assertRaisesRegex(ValueError, 'not installed'):
            validate_workflow_schema(self.graph, self.registry)
        self.assertTrue(any('required input' in error for error in workflow_schema_errors(self.graph, self.registry)))

    def test_bad_links_unknown_inputs_and_choices(self):
        for key, value, message in [('model', ['gone', 0], 'missing node'),
            ('model', ['1', 99], 'no output'), ('model', ['1', 1], 'linked output is INT'),
            ('typo', 3, 'not declared'), ('sampler', 'typo', 'not an available choice'),
            ('model', 'literal', 'requires a linked')]:
            with self.subTest(key=key, value=value):
                graph = copy.deepcopy(self.graph)
                graph['2']['inputs'][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_workflow_schema(graph, self.registry)

    def test_ranges_and_literal_types(self):
        for key, value in [('steps', True), ('steps', 20.5), ('steps', 0), ('cfg', float('nan')),
            ('cfg', float('inf')), ('cfg', -1), ('enabled', 1), ('text', None)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                graph = copy.deepcopy(self.graph)
                graph['2']['inputs'][key] = value
                validate_workflow_schema(graph, self.registry)

    def test_wildcards_unions_and_hidden_inputs(self):
        self.graph['2']['inputs']['model'] = ['1', 2]
        validate_workflow_schema(self.graph, self.registry)
        self.registry['Source']['output'][2] = 'INT,MODEL'
        validate_workflow_schema(self.graph, self.registry)
        self.registry['Target']['input']['hidden'] = {'loop_value': ['*']}
        self.graph['2']['inputs']['loop_value'] = ['1', 0]
        validate_workflow_schema(self.graph, self.registry)

    def test_invalid_or_empty_registry_and_workflow(self):
        for graph, registry in [({}, self.registry), ([], self.registry), (self.graph, {}),
            ({'1': None}, self.registry), ({'1': {'class_type': 'Source', 'inputs': []}}, self.registry)]:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_workflow_schema(graph, registry)

    def test_autogrow_names_empty_optional_refs_required_slots_and_links(self):
        self.registry['Source']['output'].append('IMAGE')
        self.registry['Target']['input']['required']['images'] = ['COMFY_AUTOGROW_V3', {'template': {
            'input': {'required': {'image': ['IMAGE', {}]}}, 'names': ['image_1', 'image_2'], 'min': 0}}]
        original = copy.deepcopy(self.registry)
        validate_workflow_schema(self.graph, self.registry)
        self.graph['2']['inputs']['images.image_1'] = ['1', 3]
        validate_workflow_schema(self.graph, self.registry)
        self.assertEqual(self.registry, original)
        self.graph['2']['inputs']['images.image_1'] = ['1', 0]
        with self.assertRaisesRegex(ValueError, 'expected IMAGE'):
            validate_workflow_schema(self.graph, self.registry)
        del self.graph['2']['inputs']['images.image_1']
        template = self.registry['Target']['input']['required']['images'][1]['template']
        template['min'] = 1
        with self.assertRaisesRegex(ValueError, 'images.image_1: required'):
            validate_workflow_schema(self.graph, self.registry)
        self.graph['2']['inputs']['images.image_1'] = ['1', 3]
        self.graph['2']['inputs']['images.image_3'] = ['1', 3]
        with self.assertRaisesRegex(ValueError, 'images.image_3: input is not declared'):
            validate_workflow_schema(self.graph, self.registry)

    def test_autogrow_prefix_and_optional_template_minimum(self):
        spec = ['COMFY_AUTOGROW_V3', {'template': {'input': {'optional': {'item': ['MODEL', {}]}},
                'prefix': 'model_', 'min': 1, 'max': 2}}]
        self.registry['Target']['input']['required']['models'] = spec
        validate_workflow_schema(self.graph, self.registry)
        self.graph['2']['inputs']['models.model_0'] = ['1', 0]
        validate_workflow_schema(self.graph, self.registry)
        spec[1]['template']['max'] = 1000
        with self.assertRaisesRegex(ValueError, 'Autogrow schema is malformed'):
            validate_workflow_schema(self.graph, self.registry)

    def test_client_preflight_reads_once_without_queueing_or_loading_models(self):
        from toolkit.comfy_sample import ComfyApiClient
        client = ComfyApiClient(timeout=1800)
        with patch.object(client, '_request_json', return_value=self.registry) as request:
            client.validate_workflow_schema(self.graph)
            client.validate_workflow_schema(self.graph)
            request.assert_called_once_with('GET', '/object_info', timeout=15)
            self.graph['1']['class_type'] = 'Missing'
            with self.assertRaisesRegex(ValueError, 'not installed'):
                client.validate_workflow_schema(self.graph)
            self.assertEqual(request.call_count, 1)

    def test_missing_edit_helper_reports_installation_without_queueing(self):
        from toolkit.comfy_sample import ComfyApiClient
        client = ComfyApiClient()
        with patch.object(client, '_request_json', return_value=self.registry) as request:
            with self.assertRaisesRegex(ValueError, 'comfy_nodes/ai_toolkit_krea_preview'):
                client.validate_workflow_schema({'1': {'class_type': 'AIToolkitKreaEditConditioning', 'inputs': {}}})
            request.assert_called_once_with('GET', '/object_info', timeout=15)

    def test_upload_receipts_allow_only_exact_load_image_paths(self):
        registry = {'LoadImage': {'input': {'required': {'image': [[], {'image_upload': True}]}},
                                 'output': ['IMAGE', 'MASK']}}
        path = 'ai-toolkit/test/reference.png'
        graph = {'1': {'class_type': 'LoadImage', 'inputs': {'image': path}}}
        with self.assertRaisesRegex(ValueError, 'not an available choice'):
            validate_workflow_schema(graph, registry)
        validate_workflow_schema(graph, registry, uploaded_images={path})
        for value in ('missing.png', '../reference.png', [path]):
            graph['1']['inputs']['image'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_workflow_schema(graph, registry, uploaded_images={path})
        graph['1']['inputs']['image'] = path
        registry['OtherLoader'] = registry['LoadImage']
        graph['1']['class_type'] = 'OtherLoader'
        with self.assertRaises(ValueError):
            validate_workflow_schema(graph, registry, uploaded_images={path})
        self.graph['1']['inputs']['name'] = path
        with self.assertRaises(ValueError):
            validate_workflow_schema(self.graph, self.registry, uploaded_images={path})

    def test_client_upload_receipts_do_not_leak_to_another_client(self):
        from toolkit.comfy_sample import ComfyApiClient
        client = ComfyApiClient()
        registry = {'LoadImage': {'input': {'required': {'image': [[], {'image_upload': True}]}},
                                 'output': ['IMAGE', 'MASK']}}
        path = 'ai-toolkit/reference.png'
        graph = {'1': {'class_type': 'LoadImage', 'inputs': {'image': path}}}
        client._uploaded_images.add(path)
        with patch.object(client, '_request_json', return_value=registry):
            client.validate_workflow_schema(graph)
        other = ComfyApiClient()
        with patch.object(other, '_request_json', return_value=registry), self.assertRaises(ValueError):
            other.validate_workflow_schema(graph)


if __name__ == '__main__':
    unittest.main()
