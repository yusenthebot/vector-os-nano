"""L19 — Detector fully removed verification tests."""
import pytest
import os


class TestDetectorRemovedFromCli:
    """Verify detector is fully removed from CLI."""

    def test_no_detector_ref_in_cli(self):
        """cli.py should have NO _detector references."""
        cli_path = os.path.expanduser(
            "~/Desktop/vector_os_nano/vector_os_nano/vcli/cli.py"
        )
        with open(cli_path) as f:
            content = f.read()
        assert "_detector" not in content, "cli.py still has _detector references"
        assert "object_detector" not in content
        assert "detect_and_project" not in content


class TestDetectorRemovedFromSimTool:
    """Verify detector is fully removed from sim_tool.py."""

    def test_no_detector_ref_in_sim_tool(self):
        """sim_tool.py should have NO _detector references."""
        path = os.path.expanduser(
            "~/Desktop/vector_os_nano/vector_os_nano/vcli/tools/sim_tool.py"
        )
        with open(path) as f:
            content = f.read()
        assert "_detector" not in content, "sim_tool.py still has _detector references"
        assert "object_detector" not in content


class TestDetectorRemovedFromAgent:
    """Verify detector service injection removed from agent.py."""

    def test_no_detector_service_injection(self):
        """agent.py should NOT inject detector into services."""
        path = os.path.expanduser(
            "~/Desktop/vector_os_nano/vector_os_nano/core/agent.py"
        )
        with open(path) as f:
            content = f.read()
        assert 'services["detector"]' not in content


class TestDetectorModuleDeleted:
    """Verify object_detector.py is deleted."""

    def test_object_detector_file_gone(self):
        """object_detector.py should NOT exist."""
        path = os.path.expanduser(
            "~/Desktop/vector_os_nano/vector_os_nano/perception/object_detector.py"
        )
        assert not os.path.isfile(path), "object_detector.py should be deleted"

    def test_detector_test_files_gone(self):
        """Detector test files should be deleted."""
        test_dir = os.path.expanduser("~/Desktop/vector_os_nano/tests/harness/")
        for f in os.listdir(test_dir):
            assert "detection" not in f.lower() or f == "test_level19_detector_cleanup.py", \
                f"Detector test file still exists: {f}"


class TestNoDetectorInActiveSkills:
    """Verify no Go2 skill uses detector."""

    SKILL_DIR = os.path.expanduser(
        "~/Desktop/vector_os_nano/vector_os_nano/skills/go2/"
    )

    def test_no_detector_in_skills(self):
        for f in os.listdir(self.SKILL_DIR):
            if not f.endswith('.py') or f == '__init__.py':
                continue
            with open(os.path.join(self.SKILL_DIR, f)) as fh:
                content = fh.read()
            assert "object_detector" not in content, f"{f} refs object_detector"
            assert "detect_and_project" not in content, f"{f} refs detect_and_project"
            assert 'services["detector"]' not in content, f"{f} uses detector service"
