#!/usr/bin/env python3
import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("STDOUT:")
        print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run AUREON test suite")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--lint", action="store_true", help="Run linting")
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not any(
        [
            args.unit,
            args.integration,
            args.coverage,
            args.lint,
            args.type_check,
            args.all,
        ]
    ):
        args.all = True

    success = True

    if args.lint or args.all:
        success &= run_command(
            ["flake8", "aureon/", "tests/", "scripts/"], "Linting with flake8"
        )
        success &= run_command(
            ["black", "--check", "aureon/", "tests/", "scripts/"],
            "Code formatting check with black",
        )

    if args.type_check or args.all:
        success &= run_command(["mypy", "aureon/"], "Type checking with mypy")

    if args.unit or args.all:
        cmd = ["pytest", "tests/", "-m", "unit"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Unit tests")

    if args.integration or args.all:
        cmd = ["pytest", "tests/", "-m", "integration"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Integration tests")

    if args.coverage or args.all:
        cmd = [
            "pytest",
            "tests/",
            "--cov=aureon",
            "--cov-report=html",
            "--cov-report=term",
        ]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "Tests with coverage")

    if not args.coverage and not args.all:
        cmd = ["pytest", "tests/"]
        if args.verbose:
            cmd.append("-v")
        success &= run_command(cmd, "All tests")

    print(f"\n{'='*60}")
    if success:
        print("🎉 All checks passed successfully!")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
