"""KTO surrogate math only; does not certify a usable training workflow."""
import unittest

import torch

from toolkit.flow_kto import FlowKTOSettings, flow_kto_terms, kto_reference_point


class FlowKTOMathTests(unittest.TestCase):
    def test_noop_losses_class_weights_and_opposite_gradient_signs(self):
        settings = FlowKTOSettings.parse({'beta': 2, 'liked_weight': 3, 'disliked_weight': 5})
        policy = torch.tensor([1., 2.], requires_grad=True)
        reference = policy.detach().clone().requires_grad_()
        loss, coefficient, point = flow_kto_terms(policy, reference, [1, 0], settings)
        torch.testing.assert_close(loss, torch.tensor([1.5, 2.5]))
        torch.testing.assert_close(coefficient, torch.tensor([1.5, -2.5]))
        self.assertEqual(point.item(), 0)
        loss.sum().backward()
        torch.testing.assert_close(policy.grad, coefficient)
        self.assertIsNone(reference.grad)

    def test_reference_point_detached_and_pooled_by_examples(self):
        first = torch.tensor([4.], requires_grad=True)
        second = torch.tensor([0., 0., 0.], requires_grad=True)
        point = kto_reference_point([first, second])
        self.assertEqual(point.item(), 1)  # NOT mean([4, 0]) = 2.
        self.assertFalse(point.requires_grad)
        self.assertEqual(kto_reference_point(torch.tensor([-2., -4.])).item(), 0)
        policy = torch.tensor([2., 1.], requires_grad=True)
        external_point = torch.tensor(1., requires_grad=True)
        loss, coefficient, _ = flow_kto_terms(policy, torch.tensor([5., 0.]), [True, False],
            FlowKTOSettings.parse(), reference_point=external_point)
        loss.sum().backward()
        torch.testing.assert_close(policy.grad, coefficient)
        self.assertIsNone(external_point.grad)

    def test_full_graph_matches_sequential_window_gradients_unequal_batches(self):
        settings = FlowKTOSettings.parse({'beta': 1.3, 'reference_estimator': 'score_window'})
        inputs = [torch.tensor([.1]), torch.tensor([.8, 1.2, -.4])]
        labels = [torch.tensor([1]), torch.tensor([0, 1, 0])]
        reference = [torch.tensor([.8]), torch.tensor([.3, .9, 1.1])]
        parameter = torch.tensor(.25, requires_grad=True)
        errors = [(values + parameter).square() for values in inputs]
        point = kto_reference_point([ref - error for ref, error in zip(reference, errors)])
        losses = [flow_kto_terms(error, ref, label, settings, reference_point=point)[0]
                  for error, ref, label in zip(errors, reference, labels)]
        torch.cat(losses).mean().backward()
        expected_gradient = parameter.grad.clone()
        replay = parameter.detach().clone().requires_grad_()
        with torch.no_grad():
            score_errors = [(values + replay).square() for values in inputs]
            replay_point = kto_reference_point([ref - error for ref, error in zip(reference, score_errors)])
            coefficients = [flow_kto_terms(error, ref, label, settings, reference_point=replay_point)[1]
                            for error, ref, label in zip(score_errors, reference, labels)]
        for values, coefficient in zip(inputs, coefficients):
            (coefficient * (values + replay).square()).sum().div(4).backward()
        torch.testing.assert_close(replay.grad, expected_gradient)

    def test_single_image_single_label_and_saturation_are_finite(self):
        for label in (0, 1):
            policy = torch.tensor([1.], requires_grad=True)
            loss, coefficient, _ = flow_kto_terms(policy, torch.tensor([2.]), [label], FlowKTOSettings.parse())
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            torch.testing.assert_close(policy.grad, coefficient)
        loss, coefficient, _ = flow_kto_terms(torch.tensor([1e6]), torch.tensor([0.]), [0], FlowKTOSettings.parse())
        self.assertTrue(torch.isfinite(loss).all() and torch.isfinite(coefficient).all())

    def test_bad_configuration_shapes_and_labels_rejected(self):
        for values in ({'beta': 0}, {'beta': True}, {'liked_weight': float('nan')},
                       {'disliked_weight': -1}, {'reference_estimator': 'ema'},
                       {'score_window_size': False}, {'unknown': 1},
                       {'reference_estimator': 'score_window', 'score_window_size': 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FlowKTOSettings.parse(values)
        for labels in ([-1], [2], [], [1, 0]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                flow_kto_terms(torch.ones(1), torch.ones(1), labels, FlowKTOSettings.parse())
        with self.assertRaises(ValueError):
            kto_reference_point([])


if __name__ == '__main__':
    unittest.main()
