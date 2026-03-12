from nls_pipeline.io import load_yaml, resolve_from_project


def test_cohort_urls_present() -> None:
    for name in ["nlsy79", "nlsy97", "cnlsy"]:
        cfg = load_yaml(resolve_from_project(f"config/{name}.yml"))
        assert cfg["download_url"].startswith("https://")
        assert cfg["raw_zip_name"].endswith(".zip")
