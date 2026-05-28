from mdpcf.vae.train_black_box import main as train_black_box
from mdpcf.vae.test_black_box import main as test_black_box


def main(args):
    print("Running black-box training/check step...")
    train_black_box(args)
    print("\nRunning black-box test step...")
    test_black_box(args)
