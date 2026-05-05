"""test_examples.py — tests for built-in examples."""

from bvp_continuation.examples import get_example_names, load_example


def test_all_examples_have_valid_structure():
    for name in get_example_names():
        problem = load_example(name)
        problem.validate_basic_structure()
