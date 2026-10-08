from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_formal_runners_have_their_summary_dependencies():
    for runner, helper in (
        ("run_final_main_ablations.ps1", "summarize_multiseed_ablations.py"),
        ("run_final_baselines.ps1", "collect_evaluation_summaries.py"),
    ):
        content = (ROOT / "scripts" / runner).read_text(encoding="utf-8")
        assert helper in content
        assert (ROOT / "scripts" / helper).is_file()
        assert '[int[]]$Seeds = @(42, 43, 44)' in content
        assert '[int]$BlackBoxSeed = 42' in content
        assert '[int]$MaxIter = 3000' in content


def test_quickstart_isolated_outputs_and_all_four_stages():
    content = (ROOT / "scripts/run_quickstart.ps1").read_text(encoding="utf-8")
    assert '"quickstart_e$Epochs"' in content
    assert '"results/quickstart/e$($Epochs)_n$NumSamples"' in content
    assert '"--method", "black_box", "--mode", "train"' in content
    assert '"--method", "csvae", "--mode", "train"' in content
    assert '"--mode", "sample"' in content
    assert '"--mode", "evaluate"' in content


def test_readme_links_only_published_quickstart_documents():
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "README.zh-CN.md" in content
    assert "docs/GITHUB_RELEASE.md" not in content
    assert "PROJECT_GUIDE.md" not in content
