import pathlib

def test_run_tf_contains_config():
    content = pathlib.Path('run_tf.py').read_text()
    assert 'class Config' in content

def test_status_constants_present():
    content = pathlib.Path('run_tf.py').read_text()
    for status in ['FALL_DETECTED', 'NORMAL', 'COLLECTING']:
        assert status in content
