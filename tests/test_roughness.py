"""Test module for roughness."""

from roughness import __author__, __email__, __version__


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Christian Tai Udovicic"
    assert __email__ == "cj.taiudovicic@gmail.com"
    assert __version__ == "0.0.0"
