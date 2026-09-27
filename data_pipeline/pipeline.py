import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
DB_FILE = "data_pipeline/zepto_books.db"


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def scrape_books():
    books = []

    for page in range(1, 6):
        url = BASE_URL + f"catalogue/page-{page}.html"

        response = requests.get(url, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for article in soup.select("article.product_pod"):
            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            rating_tag = article.select_one(".star-rating")
            availability_tag = article.select_one(".availability")

            title = title_tag.get("title", "").strip()
            price = price_tag.get_text(strip=True)
            rating_text = rating_tag.get("class", ["", "One"])[1]
            availability = availability_tag.get_text(" ", strip=True)

            books.append(
                {
                    "title": title,
                    "price": price,
                    "star_rating": rating_text,
                    "availability": availability,
                }
            )

    return pd.DataFrame(books)


def clean_data(df):
    df["price_gbp"] = (
        df["price"]
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )

    df["rating"] = df["star_rating"].map(RATING_MAP)

    df["in_stock"] = df["availability"].str.contains(
        "In stock",
        case=False,
        na=False
    )

    # Handle unexpected numeric parsing failures using median.
    df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    df["rating"] = df["rating"].fillna(df["rating"].median()).astype(int)

    # Drop rows where essential text information is unavailable.
    df = df.dropna(subset=["title"])

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
        ]
    ]


def create_database(df):
    connection = sqlite3.connect(DB_FILE)
    cursor = connection.cursor()

    cursor.execute("PRAGMA foreign_keys = ON")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    cursor.execute("DELETE FROM books")
    cursor.execute("DELETE FROM categories")

    # The assignment permits the first 5 catalogue pages.
    # These pages contain books from multiple categories.
    df["category"] = "Catalogue"

    categories = df["category"].drop_duplicates()

    for category in categories:
        cursor.execute(
            "INSERT INTO categories (category_name) VALUES (?)",
            (category,)
        )

    category_id = cursor.execute(
        "SELECT category_id FROM categories WHERE category_name = 'Catalogue'"
    ).fetchone()[0]

    for _, row in df.iterrows():
        cursor.execute(
            """
            INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                int(row["in_stock"]),
                category_id,
            ),
        )

    connection.commit()
    return connection


def run_queries(connection):
    queries = {
        "SELECT_WHERE": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp > 20;
        """,

        "ORDER_BY": """
            SELECT title, rating
            FROM books
            ORDER BY rating DESC;
        """,

        "LIMIT": """
            SELECT title, price_inr
            FROM books
            LIMIT 10;
        """,

        "DISTINCT": """
            SELECT DISTINCT rating
            FROM books
            ORDER BY rating;
        """,

        "BETWEEN": """
            SELECT title, price_gbp
            FROM books
            WHERE price_gbp BETWEEN 10 AND 30;
        """,

        "JOIN": """
            SELECT
                b.title,
                b.rating,
                c.category_name
            FROM books b
            JOIN categories c
                ON b.category_id = c.category_id
            ORDER BY b.rating DESC
            LIMIT 10;
        """,
    }

    for name, query in queries.items():
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)

        result = pd.read_sql_query(query, connection)

        print(query.strip())
        print(result.to_string(index=False))

    return queries


def main():
    print("Starting Zepto Data Pipeline...")

    raw_df = scrape_books()

    print(f"\nScraped rows: {len(raw_df)}")

    clean_df = clean_data(raw_df)

    print(f"Clean rows: {len(clean_df)}")

    if len(clean_df) < 60:
        raise ValueError(
            f"Pipeline produced only {len(clean_df)} rows. "
            "At least 60 are required."
        )

    connection = create_database(clean_df)

    queries = run_queries(connection)

    print("\n" + "=" * 70)
    print("PANDAS JOIN REPRODUCTION")
    print("=" * 70)

    books_df = pd.read_sql_query(
        "SELECT * FROM books",
        connection
    )

    categories_df = pd.read_sql_query(
        "SELECT * FROM categories",
        connection
    )

    sql_join = pd.read_sql_query(
        queries["JOIN"],
        connection
    )

    pandas_join = (
        books_df.merge(
            categories_df,
            on="category_id",
            how="inner"
        )
        [["title", "rating", "category_name"]]
        .sort_values("rating", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    print("\nSQL JOIN result:")
    print(sql_join.to_string(index=False))

    print("\nPandas merge result:")
    print(pandas_join.to_string(index=False))

    connection.close()

    print("\nPipeline completed successfully.")
    print(f"Database created at: {DB_FILE}")


if __name__ == "__main__":
    main()