import glob
import pandas as pd
import matplotlib.pyplot as plt


def cleanup(filename):
    data1 = pd.read_csv(filename, index_col=0)
    data = data1.iloc[3:, :-9].T

    data.T.to_csv("{}\\cleaned_{}".format(*filename.split("\\")), index=False)

    percent_missing = data.isnull().sum() * 100 / len(data)
    missing_value_df = pd.DataFrame({'column_name': data.columns,
                                     'percent_missing': percent_missing})
    m = missing_value_df.sort_values('percent_missing')
    m.plot(x="column_name", y="percent_missing")

    plt.title("{} missing data per feature".format(filename))
    plt.savefig(".".join(filename.split(".")[:-1] + ["jpg"]))


def clean_all():
    print("Cleaning:")
    for filename in glob.glob("Data/*.csv"):
        print("cleaning {}...".format(filename))
        cleanup(filename)
        print("cleaning done")
        print()
    print("All Clean")


if __name__ == '__main__':
    clean_all()
