import pytest, os
import numpy as np
import openfdcm

def make_rotation(line_angle):
    """
    Creates a 2D rotation matrix for the given angle in radians.

    Parameters:
    angle (float): Angle in radians for the rotation.

    Returns:
    np.ndarray: A 2x2 rotation matrix.
    """
    sin = np.sin(line_angle)
    cos = np.cos(line_angle)
    return np.array([[cos, -sin], [sin, cos]])

def create_lines(line_number, length):
    """
    Generates an array of lines with their endpoints calculated using a logarithmic space of angles to avoid symmetries.

    Parameters:
    line_number (int): Number of lines to generate.
    length (float): The length of each line.

    Returns:
    np.ndarray: A 4 x line_number array where each column represents a line.
                The first two rows represent the starting point (always [0, 0]),
                and the last two rows represent the endpoint of each line.
    """
    line_array = np.zeros((4, line_number))
    for i, line_angle in enumerate(np.logspace(np.log10(2 * np.pi), np.log10(4 * np.pi), line_number)):
        rotation_matrix = make_rotation(line_angle)
        endpoint = np.matmul(rotation_matrix, np.array([length, 0]))
        line_array[:, i] = np.array([0, 0, endpoint[0], endpoint[1]])
    return line_array

def all_close(a, b, atol=1e-5):
    return np.allclose(a, b, atol=atol)

def apply_transform(lines, transform):
    return (np.matmul(transform[:2, :2], lines.reshape(2, -1)) + transform[:2, 2:3]).reshape(4, -1)

def run_test(scene_ratio, num_threads):
    max_tmpl_lines, max_scene_lines = 4, 10
    coeff = 5.0
    depth = 30

    threadpool = openfdcm.ThreadPool(num_threads)
    search_strategy = openfdcm.DefaultSearch(max_tmpl_lines, max_scene_lines)
    optimizer_strategy = openfdcm.DefaultOptimize(threadpool)
    matcher = openfdcm.DefaultMatch()
    penalizer = openfdcm.ExponentialPenalty(1.5)
    number_of_lines = 10
    line_length = 100
    tmpl = create_lines(number_of_lines, line_length)

    scene_transform = np.array([[-1, 0, line_length], [0, -1, line_length]])
    scene = apply_transform(tmpl, scene_transform)

    for distance in [openfdcm.distance.L2, openfdcm.distance.L1, openfdcm.distance.L2_SQUARED]:
        featuremap_params = openfdcm.Dt3CpuParameters(depth=depth, dt3Coeff=coeff,
                                                      padding=2.2, distance=distance)
        featuremap = openfdcm.build_cpu_featuremap(scene, featuremap_params, threadpool)
        raw_matches = openfdcm.search(matcher, search_strategy, optimizer_strategy, featuremap, [tmpl], scene)
        sorted_matches = openfdcm.sort_matches(raw_matches)

        best_match_transform = sorted_matches[0].transform
        best_match_rotation = best_match_transform[:2, :2]
        best_match_translation = best_match_transform[:2, 2]

        assert len(sorted_matches) == min(max_tmpl_lines, number_of_lines) * min(number_of_lines, max_scene_lines) * 2
        assert all_close(scene_transform[:2, :2], best_match_rotation)
        assert all_close(scene_transform[:2, 2], best_match_translation, 1e0 * 1 / scene_ratio)

        scene_transform = np.array([[1, 0, 0], [0, 1, 0]])
        scene = apply_transform(tmpl, scene_transform)
        featuremap = openfdcm.build_cpu_featuremap(scene, featuremap_params, threadpool)
        raw_matches = openfdcm.search(matcher, search_strategy, optimizer_strategy, featuremap, [tmpl], scene)
        penalized_matches = openfdcm.penalize(penalizer, raw_matches, openfdcm.get_template_lengths([tmpl]))
        sorted_matches = openfdcm.sort_matches(penalized_matches)

        best_match_rotation = sorted_matches[0].transform[:2, :2]
        best_match_translation = sorted_matches[0].transform[:2, 2]

        assert len(raw_matches) == max_tmpl_lines * max_scene_lines * 2
        assert all_close(scene_transform[:2, :2], best_match_rotation)
        assert all_close(scene_transform[:2, 2], best_match_translation, 1e0 * 1 / scene_ratio)

        scene = np.zeros((4, 0))
        featuremap = openfdcm.build_cpu_featuremap(scene, featuremap_params, threadpool)
        matches = openfdcm.search(matcher, search_strategy, optimizer_strategy, featuremap, [tmpl], scene)
        assert len(matches) == 0

        templates = []
        scene = tmpl
        featuremap = openfdcm.build_cpu_featuremap(scene, featuremap_params, threadpool)
        matches = openfdcm.search(matcher, search_strategy, optimizer_strategy, featuremap, [], scene)
        assert len(matches) == 0

        templates = [np.zeros((4, 0))]
        matches = openfdcm.search(matcher, search_strategy, optimizer_strategy, featuremap, templates, scene)
        assert len(matches) == 0

@pytest.mark.parametrize("scene_ratio", [1.0, 0.3])
@pytest.mark.parametrize("num_threads", [4])
def test_matching(scene_ratio, num_threads):
    run_test(scene_ratio, num_threads)

def test_write_read():
    lines = create_lines(100, 10)
    filepath: str = "./test_write_array.lines"
    openfdcm.write(filepath, lines)
    try:
        read_lines = openfdcm.read(filepath)
        os.remove(filepath)
        assert all_close(lines, read_lines)
    except:
        os.remove(filepath)


if __name__ == "__main__":
    pytest.main()
