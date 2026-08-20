from medical_opd.cli import build_parser


def test_public_cli_dispatches_clean_commands() -> None:
    parser = build_parser()

    prepare = parser.parse_args(["prepare-medical-sft-data"])
    sft = parser.parse_args(
        ["plan-medical-sft", "--steps", "1", "--output-dir", "runs/medical-sft"]
    )
    opd = parser.parse_args(
        [
            "plan-staged-opd",
            "--stage",
            "medical",
            "--steps",
            "25",
            "--output-dir",
            "runs/medical-opd",
        ]
    )

    assert prepare.command == "prepare-medical-sft-data"
    assert sft.command == "plan-medical-sft"
    assert opd.command == "plan-staged-opd"
