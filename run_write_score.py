import sys
from utils import write_score_file

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_write_score.py <task> <save_path>")
        sys.exit(1)

    task = sys.argv[1]
    save_path = sys.argv[2]

    write_score_file(task, save_path)

if __name__ == "__main__":
    main()
