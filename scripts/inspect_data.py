from betproj.data_loader import (
    load_closing_odds,
    load_odds_series_b,
    load_odds_series_b_matches,
)


def show(name, df, n_cols=80):
    print("\n" + "=" * 100)
    print(name)
    print("shape:", df.shape)
    print("first columns:")
    for col in df.columns[:n_cols]:
        print(col)
    print("\nhead:")
    print(df.head(3))


def main():
    show("closing_odds", load_closing_odds(), n_cols=120)
    show("odds_series_b", load_odds_series_b(), n_cols=120)
    show("odds_series_b_matches", load_odds_series_b_matches(), n_cols=120)


if __name__ == "__main__":
    main()