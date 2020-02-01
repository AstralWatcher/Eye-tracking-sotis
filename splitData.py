if __name__ == "__main__":
    import pandas as pd

    print("Starting to cut")
    path = "C:\Users\Astral\Desktop\Master\Sotis\sotisDataToCut"

    df = pd.read_excel(io=path) # sheet_name=sheet)
    print(df.head(5))  # print first 5 rows of the dataframe
