import pandas as pd


def load_csv(file):
    """
    Load a CSV file into a pandas DataFrame.
    """

    try:
        df = pd.read_csv(
            file,
            encoding="latin1"
        )

        return df

    except Exception as e:
        raise Exception(f"Unable to load CSV file: {e}")