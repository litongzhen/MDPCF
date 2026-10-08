from scripts.run_generation_strategy_probe import strategy_overrides


def test_strategies_are_scoped_and_leave_reference_config_unchanged():
    config = {"latent_init_radius_scale": .2, "proximity_weight_input": 1.0}
    strategies = strategy_overrides(config)
    assert strategies["reference"] == {}
    assert strategies["radius_half"] == {"latent_init_radius_scale": .1}
    assert strategies["proximity_double"] == {"proximity_weight_input": 2.0}
    assert strategies["trajectory"] == {"trajectory_quality_selection": True}
    assert config == {"latent_init_radius_scale": .2, "proximity_weight_input": 1.0}
