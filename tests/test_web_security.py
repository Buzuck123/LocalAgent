def test_api_version_matches_version_module():
    from agent import web
    from agent.version import __version__
    assert web.version() == {"name": "LocalAgent", "version": __version__}
    assert web.app.version == __version__
