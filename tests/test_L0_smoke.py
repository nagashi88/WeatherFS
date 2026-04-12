import pytest


@pytest.mark.L0
class TestL0Smoke:
    def test_all_modules_importable(self):
        """Every module in the system can be imported without error."""
        import weatheredge.config
        import weatheredge.data.weather
        import weatheredge.data.noaa
        import weatheredge.data.market
        import weatheredge.probability.engine
        import weatheredge.probability.gaussian
        import weatheredge.probability.ensemble
        import weatheredge.probability.edge
        import weatheredge.probability.bias_correction
        import weatheredge.probability.bucket
        import weatheredge.agents.risk_manager
        import weatheredge.agents.portfolio_manager
        import weatheredge.agents.qualitative
        import weatheredge.agents.base
        import weatheredge.agents.strategies.conservative
        import weatheredge.agents.strategies.aggressive
        import weatheredge.agents.strategies.mean_reversion
        import weatheredge.execution.paper_trader
        import weatheredge.execution.position_tracker
        import weatheredge.execution.fill_monitor
        import weatheredge.execution.cross_platform
        import weatheredge.notifications.telegram
        import weatheredge.notifications.formatters
        import weatheredge.learning.calibration
        import weatheredge.learning.resolution_tracker
        import weatheredge.learning.weight_updater
        import weatheredge.learning.agent_weight_updater
        import weatheredge.learning.rmse_updater
        import weatheredge.learning.error_analysis
        import weatheredge.intelligence.wallet_profiler
        import weatheredge.intelligence.edge_decay_monitor
        import weatheredge.intelligence.competitor_tracker
        import weatheredge.backtest.engine
        import weatheredge.backtest.metrics
        import weatheredge.llm

    def test_db_creation(self, tmp_path):
        """SQLite DB can be created with all tables in a temp directory."""
        from weatheredge.db.database import make_engine, init_db
        db_path = tmp_path / "test.db"
        engine = make_engine(f"sqlite:///{db_path}")
        init_db(engine)
        from sqlalchemy import inspect
        insp = inspect(engine)
        tables = insp.get_table_names()
        assert len(tables) >= 11  # At least 11 original tables
        engine.dispose()

    def test_config_loading(self):
        """Settings loads with defaults without error."""
        from weatheredge.config import Settings
        # Ignore local .env so defaults remain deterministic in CI/local runs.
        s = Settings(_env_file=None)
        assert s.execution_mode == "paper"
        assert s.initial_bankroll == 1000.0
        assert s.kelly_fraction == 0.08  # PRD v3.1: reduced from 0.15

    def test_scheduler_starts_and_stops(self):
        """APScheduler can start, add a job, and shut down cleanly."""
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(lambda: None, 'interval', seconds=60, id='test_job')
        scheduler.start()
        assert scheduler.running
        scheduler.shutdown(wait=False)
        assert not scheduler.running
