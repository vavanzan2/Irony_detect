#comment

import argparse

import step1_load
import step2_aggregate
import step3_split
import step4_baseline
import step5_cardiff
import step6_finetune
import step7_evaluate
import step8_compare


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline — Detection of Irony with EPIC",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[
            "all", "load", "aggregate", "split",
            "baseline", "cardiff", "finetune", "evaluate", "compare",
        ],
        default=["all"],
        help=(
            "Pipeline steps to execute.\n"
            "Use 'all' for complete pipeline.\n"
            "Example: --steps load aggregate split"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory where the results are saved. (default: outputs/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility. (default: 42)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs for RoBERTa. (default: 3)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size used during fine-tuning. (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate used during fine-tuning. (default: 2e-5)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Classification threshold (probability of the 'irony' class) for the fine-tuned RoBERTa model during the evaluation step. (default: 0.3)")
    return parser.parse_args()


def expand_steps(steps):
    all_steps = [
        "load", "aggregate", "split",
        "baseline", "cardiff",
        "finetune", "evaluate", "compare",
    ]
    return all_steps if "all" in steps else steps


def main():
    args = parse_args()
    steps = expand_steps(args.steps)

    print("=" * 60)
    print("  Pipeline — Detection of Irony with EPIC")
    print("=" * 60)
    print(f"  Selected steps : {steps}")
    print(f"  Output directory  : {args.output_dir}")
    print(f"  Seed                : {args.seed}")
    print("=" * 60)

    raw_df = labeled_df = None
    train_df = val_df = test_df = None

    if "load" in steps:
        raw_df = step1_load.run(output_dir=args.output_dir)

    if "aggregate" in steps:
        labeled_df = step2_aggregate.run(raw_df=raw_df, output_dir=args.output_dir)

    if "split" in steps:
        train_df, val_df, test_df = step3_split.run(
            labeled_df=labeled_df, seed=args.seed, output_dir=args.output_dir
        )

    if "baseline" in steps:
        step4_baseline.run(train_df=train_df, test_df=test_df, output_dir=args.output_dir)

    if "cardiff" in steps:
        step5_cardiff.run(test_df=test_df, output_dir=args.output_dir)

    if "finetune" in steps:
        step6_finetune.run(
            train_df=train_df, val_df=val_df, output_dir=args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

    if "evaluate" in steps:
        step7_evaluate.run(test_df=test_df, output_dir=args.output_dir, threshold=args.threshold)

    if "compare" in steps:
        step8_compare.run(output_dir=args.output_dir)

    print("\n" + "=" * 60)
    print("  Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
