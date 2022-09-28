from pprint import pprint

import aim

if __name__ == "__main__":
    repo = aim.Repo(".")
    move_map = [
        "2s3z",
        "25m",
        "6h_vs_8z",
        "3s5z_vs_3s6z",
        "MMM",
        "10m_vs_11m",
        "corridor",
    ]
    for run in repo.iter_runs():
        try:
            if run.archived == False:
                print(run.experiment)
        except:
            continue
    repo.close()
