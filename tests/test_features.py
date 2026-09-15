"""Analytic native-encoding/gate regressions; oracle accuracy is scored separately."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from src import features as f


def control_to_power(cp):
    """Independent Bernstein expansion used only to construct test fixtures."""
    p0, p1, p2, p3 = np.asarray(cp, dtype=np.float64)
    return np.stack([p0, 3*(p1-p0), 3*(p0-2*p1+p2), -p0+3*p1-3*p2+p3])


class CurveFeatureTests(unittest.TestCase):
    def test_power_to_control_points_preserves_curve(self):
        omega = np.array([[2., 3., 4.], [1., -2., 3.], [-4., 5., 2.], [3., 1., -1.]])
        cp = f.power_to_control_points(omega)
        s = np.linspace(0., 1., 17)
        basis = np.stack([(1-s)**3, 3*s*(1-s)**2, 3*s*s*(1-s), s**3], axis=1)
        np.testing.assert_allclose(basis @ cp, f._eval(omega, s), atol=1e-12)

    def test_two_points_produce_a_true_line(self):
        points = np.array([[1., 2., 4.], [4., 6., 9.]])
        omega = f._solve_coeffs(points, np.array([0., 1.]))
        np.testing.assert_allclose(omega[2:], 0., atol=1e-12)
        np.testing.assert_allclose(f._eval(omega, np.array([0., 1.])), points, atol=1e-12)
        row = f.curve_features(omega, False, f.CurveSettings())
        np.testing.assert_allclose(row, [1., 3., 4., 0., 1/3, 0., 1/3, 5., 5/3, -5/3], atol=1e-6)
        self.assertEqual(row.dtype, np.float32)

    def test_native_interleaving_and_signed_angles(self):
        omega = control_to_power([[0, 0, 0], [2, 1, 2], [-1, 2, 4], [1, 0, 8]])
        row = f.curve_features(omega, False, f.CurveSettings())
        np.testing.assert_allclose(row, [1, 1, 0, np.arctan2(1, 2), np.sqrt(5),
                                         -np.pi/4, np.sqrt(8), 8, 2, -4], atol=1e-6)
        # Fixed endpoint pairing, and ratios > 1 are not clipped for en-US.
        self.assertGreater(row[4], 1)
        self.assertGreater(row[6], 1)

    def test_straight_angles_zero_in_every_quadrant(self):
        for dx, dy in [(1., 0.), (0., 1.), (-1., 0.), (0., -1.), (-1., -1.)]:
            with self.subTest(dx=dx, dy=dy):
                omega = np.zeros((4, 3))
                omega[1] = [dx, dy, np.hypot(dx, dy)]
                row = f.curve_features(omega, True, f.CurveSettings())
                np.testing.assert_allclose(row[3:7], [0., 1/3, 0., 1/3], atol=1e-6)
                self.assertEqual(row[0], 0.)

    def test_zero_chord_does_not_override_atan2(self):
        # In the explicit native expression, the second denominator is -0.
        row = f.curve_features(np.zeros((4, 3)), False, f.CurveSettings())
        self.assertEqual(row[0], 1.)
        np.testing.assert_array_equal(row[[4, 6]], [0., 0.])
        self.assertEqual(row[5], np.arctan2(np.float32(0), np.float32(-0.)))
        self.assertTrue(np.isfinite(row).all())

    def test_tiny_positive_chord_still_divides(self):
        omega = np.zeros((4, 3))
        omega[1, 0] = 1e-10
        row = f.curve_features(omega, False, f.CurveSettings())
        np.testing.assert_allclose(row[[4, 6]], [1/3, 1/3], atol=1e-6)

    def test_time_legs_ignore_large_absolute_timestamp(self):
        omega = np.zeros((4, 3), dtype=np.float32)
        omega[:, 2] = [2**30, 1.1234567, -2.345678, 3.456789]
        g1, g2, g3 = omega[1:, 2]
        expected = np.array([(g1+g2)+g3, g1/np.float32(3),
                             (-g1/np.float32(3)-(np.float32(2)*g2)/np.float32(3))-g3], dtype=np.float32)
        row = f.curve_features(omega, False, f.CurveSettings())
        np.testing.assert_array_equal(row[7:].view(np.uint32), expected.view(np.uint32))
        # Absolute float32 control points lose the legs completely at this offset.
        cp = f.power_to_control_points(omega)
        np.testing.assert_array_equal(np.diff(cp[:, 2]), [0., 0., 0.])

    def test_no_time_is_seven_features_including_empty_output(self):
        cfg = f.CurveSettings(interpolate_time=False)
        self.assertEqual(f.curve_features(np.zeros((4, 3)), True, cfg).shape, (7,))
        self.assertEqual(f.extract_features([], cfg).shape, (0, 7))
        self.assertEqual(f.extract_features([f.Stroke([0, 1], [0, 1], [0, 1])], cfg).shape, (1, 7))

    def test_normalization_is_specific_to_each_feature_family(self):
        omega = control_to_power([[0, 0, 0], [2, 1, 2], [-1, 2, 4], [1, 0, 8]])
        cfg = f.CurveSettings(normalize_outputs_to_zero_one=True)
        row = f.curve_features(omega, True, cfg)
        np.testing.assert_allclose(row, [0, 1, .5, (np.arctan2(1,2)+np.pi)/(2*np.pi),
                                         1, .375, 1, 1, 1, 0], atol=1e-6)
        short = f.curve_features(omega, True, replace(cfg, interpolate_time=False))
        np.testing.assert_array_equal(short, row[:7])


class FitGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = f.CurveSettings()
        self.omega = np.zeros((4, 3))
        self.omega[1] = [1., 0., 1.]
        self.s = np.linspace(0., 1., 10)
        self.points = f._eval(self.omega, self.s)

    def failures(self, points, cfg=None, diagonal=1.):
        return f._fit_failures(points, self.omega, self.s, cfg or self.cfg, diagonal)

    def test_max_and_rms_are_separate_strict_spatial_gates(self):
        points = self.points.copy()
        points[4, 1] = .021  # RMS < .01; only max > .02.
        self.assertEqual(self.failures(points), (True, False))
        points[:, 1] = .011  # Max < .02; only RMS > .01.
        self.assertEqual(self.failures(points), (True, False))
        points[:, 1] = .01
        self.assertEqual(self.failures(points), (False, False))
        points = self.points.copy()
        points[4, 1] = .02
        self.assertEqual(self.failures(points), (False, False))
        points[:, 2] += np.arange(10)*1e9
        self.assertEqual(self.failures(points), (False, False))

    def test_gates_scale_with_whole_stroke_bbox_diagonal(self):
        points = self.points.copy()
        points[:, 1] = .015
        self.assertTrue(self.failures(points, diagonal=1.)[0])
        self.assertFalse(self.failures(points, diagonal=2.)[0])
        self.assertAlmostEqual(f._bbox_diagonal(np.array([[0, 0, 1e9], [3, 4, -1e9]])), 5.)
        calls=[]
        original=f._fit_failures
        def record(points, omega, s, cfg, diagonal):
            calls.append(diagonal)
            return original(points, omega, s, cfg, diagonal)
        points = np.column_stack([np.linspace(0, 1, 17), np.sin(np.linspace(0, 8, 17)), np.arange(17)])
        with patch.object(f, '_fit_failures', side_effect=record):
            segs=f._split_points(points, self.cfg)
            f.merge_beziers(segs, self.cfg)
        self.assertGreater(len(calls), 2)
        np.testing.assert_array_equal(calls, np.full(len(calls), f._bbox_diagonal(points)))

    def test_polygon_length_not_integrated_arc(self):
        omega = control_to_power([[0,0,0],[0,1,0],[1,1,0],[1,0,0]])
        s=np.linspace(0,1,21)
        points=f._eval(omega,s)
        self.assertLess(f._arc_length(omega), 2.5)
        self.assertEqual(f._fit_failures(points,omega,s,replace(self.cfg,max_arc_ratio=2.5),2.),(False,True))
        self.assertEqual(f._fit_failures(points,omega,s,self.cfg,2.),(False,False))

    def test_adjacent_control_leg_reversal_is_independent_gate(self):
        omega = control_to_power([[0,0,0],[1,0,0],[0,0,0],[2,0,0]])
        s=np.linspace(0,1,11)
        points=f._eval(omega,s)
        self.assertEqual(f._fit_failures(points,omega,s,self.cfg,2.),(False,True))
        self.assertEqual(f._fit_failures(points,omega,s,replace(self.cfg,split_cos_threshold=-1),2.),(False,False))

    def test_closed_loop_and_zero_legs(self):
        omega = control_to_power([[0,0,0],[1,0,0],[0,1,0],[0,0,0]])
        s=np.linspace(0,1,11)
        self.assertTrue(f._fit_failures(f._eval(omega,s),omega,s,self.cfg,2.)[1])
        for cp, rejected in [
            ([[0,0,0],[0,0,0],[.5,0,0],[1,0,0]], True),   # V=0 -> source a NaN
            ([[0,0,0],[.5,0,0],[.5,0,0],[1,0,0]], True),  # M=0 -> final 0/0
            ([[0,0,0],[.5,0,0],[1,0,0],[1,0,0]], False),  # W=0 -> ignore b NaN
            ([[0,0,0]]*4, True),                         # zero chord rejects
        ]:
            with self.subTest(cp=cp):
                omega=control_to_power(cp)
                self.assertEqual(f._fit_failures(f._eval(omega,s),omega,s,self.cfg,1.),(False,rejected))
        omega=np.full((4,3),np.nan)
        self.assertTrue(f._fit_failures(f._eval(omega,s),omega,s,self.cfg,1.)[1])

    def test_exact_three_point_reversal_bypasses_geometry(self):
        points=np.array([[0.,0.,0.],[1.,0.,1.],[0.,0.,2.]])
        self.assertEqual(len(f._split_points(points,self.cfg)),1)
        curves=f.fit_beziers(points,self.cfg)
        self.assertEqual(len(curves),1)
        np.testing.assert_array_equal(curves[0][3], [0.,0.,0.])

    def test_singleton_copies_xy_but_not_time(self):
        points=np.array([[5.,6.,12345.]])
        curves=f.fit_beziers(points,self.cfg)
        self.assertEqual(len(curves),1)
        expected=np.zeros((4,3)); expected[0,:2]=[5.,6.]
        np.testing.assert_array_equal(curves[0],expected)
        row=f.extract_features([f.Stroke([5],[6],[12345])])
        self.assertEqual(row.shape,(1,10))
        np.testing.assert_array_equal(row[0,7:], [0.,0.,0.])

    def test_corner_radius_equality_skip_and_earliest_tie(self):
        corner=np.array([[0,0,0],[1,0,0],[1,1,0]])
        self.assertEqual(f._split_at_min_angle(corner,1.),1)
        self.assertIsNone(f._split_at_min_angle(corner,1.01))
        tied=np.array([[0,0,0],[1,0,0],[1,1,0],[2,1,0]])
        self.assertEqual(f._split_at_min_angle(tied,1.),1)
        straight=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]])
        self.assertIsNone(f._split_at_min_angle(straight,.1))
        self.assertIsNone(f._split_at_min_angle(straight,10.))

    def test_missing_residual_corner_falls_through_to_geometry(self):
        points=np.array([[0.,0.,0.],[1.,0.,1.],[2.,0.,2.],[3.,0.,3.]])
        with patch.object(f, '_fit_failures', return_value=(True,True)), \
             patch.object(f, '_split_at_min_angle', return_value=None), \
             patch.object(f, '_split_at_max_curvature', return_value=1) as curvature:
            segs=f._split_points(points,self.cfg)
            self.assertEqual([len(seg) for seg in segs],[2,3])
            curvature.assert_called_once()
        with patch.object(f, '_fit_failures', return_value=(True,False)), \
             patch.object(f, '_split_at_min_angle', return_value=None), \
             patch.object(f, '_split_at_max_curvature') as curvature:
            self.assertEqual(len(f._split_points(points,self.cfg)),1)
            curvature.assert_not_called()

    def test_curvature_uses_100_interior_parameters_and_earliest_nearest(self):
        for s in (np.array([0,.2,.4,.8,1]), np.array([0,.9,.2,.4,.2,1])):
            with self.subTest(s=s):
                grids=[]
                def d1(omega,q):
                    grids.append(q.copy())
                    result=np.zeros((len(q),3),dtype=np.float32); result[:,0]=1
                    return result
                def d2(omega,q):
                    result=np.zeros((len(q),3),dtype=np.float32)
                    result[:,1]=np.arange(len(q)) # Curvature maximum is last grid sample.
                    return result
                with patch.object(f,'_eval_d1',side_effect=d1), patch.object(f,'_eval_d2',side_effect=d2):
                    index=f._split_at_max_curvature(np.zeros((len(s),3)),self.omega,s)
                self.assertEqual(len(grids[0]),100)
                self.assertEqual(grids[0].dtype,np.float32)
                self.assertEqual(grids[0][0],np.float32(s[1]))
                self.assertEqual(grids[0][-1],np.float32(s[-2]))
                self.assertEqual(index,int(np.argmin(np.abs(s[1:-1]-s[-2])))+1)

    def test_all_zero_curvature_still_selects_first_interior_point(self):
        s=np.array([0.,.2,.8,1.])
        points=f._eval(self.omega,s)
        self.assertEqual(f._split_at_max_curvature(points,self.omega,s),1)
        self.assertEqual(f._split_at_max_curvature(np.zeros((4,3)),np.zeros((4,3)),s),1)

    def test_constant_position_guard_is_finite_deliberate_divergence(self):
        points=np.tile([5.,6.,7.],(6,1))
        self.assertTrue(np.isfinite(f._initial_s(points)).all())
        self.assertTrue(np.isfinite(f.extract_features([f.Stroke(points[:,0],points[:,1],points[:,2])])).all())


class ThinningTests(unittest.TestCase):
    def points(self,x):
        return np.column_stack([x,np.zeros(len(x)),np.arange(len(x))])

    def test_dp_can_drop_first_point_to_retain_a_longer_chain(self):
        points=self.points([0,2,-2,4,10000])
        kept=f._thin_points(points)
        # Greedy starting at 0 would keep only 0,4,10000; native optimum has 4 points.
        np.testing.assert_array_equal(kept[:,2],[1,2,3,4])
        fitted=f._normalize_curve_time(kept)
        self.assertAlmostEqual(float(fitted[-1,2]-fitted[0,2]),10006,places=2)

    def test_earliest_predecessor_and_latest_endpoint_ties(self):
        np.testing.assert_array_equal(f._thin_points(self.points([0,1,5,10000]))[:,2],[0,2,3])
        np.testing.assert_array_equal(f._thin_points(self.points([0,10000,10001]))[:,2],[0,2])

    def test_strict_spacing_and_no_transition_singleton_zero(self):
        diagonal=np.float32(1)/np.float32(0.0003452669770922512)
        e=diagonal*np.float32(0.0003452669770922512)
        with patch.object(f,'_bbox_diagonal',return_value=diagonal):
            np.testing.assert_array_equal(f._thin_points(self.points([0,e,2*e]))[:,2],[0,2])
        np.testing.assert_array_equal(f._thin_points(self.points([1,1,1]))[:,2],[0])
        np.testing.assert_array_equal(f._thin_points(self.points([0,np.nan,1]))[:,2],[0])
        self.assertEqual(f._thin_points(np.empty((0,3))).shape,(0,3))

    def test_thinning_prevents_constant_position_initial_parameterization(self):
        with patch.object(f,'_initial_s',side_effect=AssertionError('singleton must bypass initial s')):
            rows=f.extract_features([f.Stroke([5,5,5],[6,6,6],[1,2,3])])
        self.assertEqual(rows.shape,(1,10))
        np.testing.assert_array_equal(rows[0,7:],[0,0,0])


class PipelineTests(unittest.TestCase):
    def test_size_leaves_time_untouched_then_fitter_scales_it(self):
        stroke = f.Stroke([10, 40], [20, 60], [100, 200])
        result = f.normalize_size(f.normalize_time([stroke]))[0]
        np.testing.assert_array_equal(result.t, [0.,100.])
        np.testing.assert_allclose(result.x, [0., .75])
        np.testing.assert_allclose(result.y, [0., 1.])
        fitted=f._normalize_curve_time(result.points)
        self.assertAlmostEqual(fitted[-1,2]-fitted[0,2],50/40)

    def test_global_origin_is_subtracted_once_without_order_repair(self):
        strokes=[f.Stroke([0,1],[0,1],[100,200]), f.Stroke([2,3],[0,1],[250,240])]
        result=f.normalize_time(strokes)
        np.testing.assert_array_equal(result[0].t,[0,100])
        np.testing.assert_array_equal(result[1].t,[150,140])
        missing=[f.Stroke([0],[0]),strokes[1]]
        self.assertIs(f.normalize_time(missing),missing)

    def test_any_timestamp_size_mismatch_regenerates_every_stroke(self):
        strokes=[f.Stroke([0,1],[0,1],[500,600]), f.Stroke([2,3,4],[0,1,2],[8])]
        result=f.hallucinate_time(strokes)
        np.testing.assert_array_equal(result[0].t,[0,20])
        np.testing.assert_array_equal(result[1].t,[40,60,80])
        self.assertIs(f.hallucinate_time(result),result)
        forced=f.hallucinate_time(result,SimpleNamespace(unknown_1=7.,unknown_2=True))
        np.testing.assert_array_equal(forced[1].t,[14,21,28])
        for ts in ([1,1],[2,1],[np.nan,0]):
            matching=[f.Stroke([0,1],[0,1],ts)]
            self.assertIs(f.hallucinate_time(matching),matching)

    def test_size_width_floor_first_point_origin_and_epsilon(self):
        flat=f.normalize_size([f.Stroke([4,204],[5,5],[10,20])])[0]
        np.testing.assert_allclose(flat.x,[0,100])
        np.testing.assert_array_equal(flat.t,[10,20])
        reverse=f.normalize_size([f.Stroke([20,0],[30,10],[10,20])])[0]
        np.testing.assert_allclose(reverse.x,[0,-1])
        tiny=f.normalize_size([f.Stroke([0,1e-9],[0,1e-9],[0,1])])[0]
        np.testing.assert_allclose(tiny.x,[0,1e-9],atol=1e-15)

    def test_fitter_rescale_preserves_offset_and_uses_spatial_length(self):
        points=np.array([[0.,0.,10.],[3.,4.,20.]])
        scaled=f._normalize_curve_time(points)
        np.testing.assert_array_equal(scaled[:,2],[5.,10.])
        np.testing.assert_array_equal(points[:,2],[10.,20.])
        for ts in ([20,20],[20,10],[np.nan,20],[10,np.nan]):
            bad=points.copy();bad[:,2]=ts
            np.testing.assert_array_equal(f._normalize_curve_time(bad)[:,2],[0,0])
        # Synthetic bridges take the same fitter rescale, with no flag exemption.
        strokes=[f.Stroke([0,0],[0,1],[100,200]),f.Stroke([1,1],[0,1],[300,400])]
        processed=f.run_pipeline(strokes)
        bridge=processed[1]
        self.assertTrue(bridge.pen_up)
        self.assertGreater(bridge.t[-1],bridge.t[0])
        scaled=f._normalize_curve_time(bridge.points)
        self.assertAlmostEqual(float(scaled[-1,2]-scaled[0,2]),np.sqrt(2),places=6)


    def test_penup_inherits_endpoint_coordinates_and_times(self):
        a = f.Stroke([0, 1], [1, 2], [0, 2])
        b = f.Stroke([3, 4], [4, 5], [5, 6])
        out = f.add_penup_strokes([a, b])
        self.assertEqual(len(out), 3)
        self.assertTrue(out[1].pen_up)
        np.testing.assert_array_equal(out[1].points, [[1, 2, 2], [3, 4, 5]])
        np.testing.assert_array_equal(out[0].points,a.points)
        np.testing.assert_array_equal(out[2].points,b.points)
        self.assertFalse(out[0].pen_up)
        self.assertFalse(out[2].pen_up)
        absent=f.add_penup_strokes([f.Stroke([0],[0],pen_up=True),f.Stroke([],[],[]),b])
        self.assertEqual(len(absent),3)
        self.assertEqual(len(absent[1].t),0)
        self.assertFalse(absent[0].pen_up)

    def test_named_pipeline_and_optional_penup(self):
        self.assertEqual(f.HANDWRITING_PIPELINE, (
            'normalize_time', 'hallucinate_time',
            'normalize_size_writing_guide_first_stroke', 'add_pen_up_strokes'))
        strokes = [f.Stroke([0, 0], [0, 1], [0, 1]), f.Stroke([1, 1], [0, 1], [2, 3])]
        before = [s.points.copy() for s in strokes]
        with_pen = f.extract_features(strokes)
        without_pen = f.extract_features(strokes, include_penup=False)
        self.assertEqual(with_pen.shape, (3, 10))
        self.assertEqual(without_pen.shape, (2, 10))
        self.assertEqual(with_pen.dtype, np.float32)
        np.testing.assert_array_equal(with_pen[:, 0], [1., 0., 1.])
        for s, original in zip(strokes, before):
            np.testing.assert_array_equal(s.points, original)
        with self.assertRaisesRegex(KeyError, 'unimplemented preprocessing step'):
            f.run_pipeline(strokes, ['not_a_step'])

    def test_empty_input(self):
        for strokes in ([], [f.Stroke([], [], [])]):
            result = f.extract_features(strokes)
            self.assertEqual(result.shape, (0, 10))
            self.assertEqual(result.dtype, np.float32)

    def test_empty_stroke_mismatch_must_reach_global_hallucination(self):
        strokes=[f.Stroke([],[],[7]),f.Stroke([0,1,2],[0,1,0],[10,11,100])]
        processed=f.run_pipeline(strokes)
        self.assertEqual(len(processed),1)
        np.testing.assert_array_equal(processed[0].t,[0,20,40])
        np.testing.assert_array_equal(f.extract_features(strokes),
                                      f.extract_features(processed,pipeline=()))
        with self.assertRaisesRegex(ValueError,r'Malformed input.*stroke=0'):
            f.extract_features(strokes,pipeline=())

    def test_processor_rejects_unrepaired_timestamp_mismatch_with_stroke_index(self):
        with self.assertRaisesRegex(ValueError,r'Malformed input.*stroke=0'):
            f.extract_features([f.Stroke([0,1],[0,1])],pipeline=())

    def test_json_optional_timestamps_and_length_validation(self):
        strokes = f.strokes_from_json({'strokes': [{'x': [0, 1], 'y': [2, 3]}]})
        self.assertEqual(len(strokes[0].t),0)
        with self.assertRaisesRegex(ValueError,'completed before fitting'):
            _=strokes[0].points
        np.testing.assert_array_equal(f.hallucinate_time(strokes)[0].t,[0,20])
        with self.assertRaisesRegex(ValueError, 'same length'):
            f.Stroke([1, 2], [1], [1, 2])


if __name__ == '__main__':
    unittest.main()
