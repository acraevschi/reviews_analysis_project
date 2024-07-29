import pandas as pd


def process_yelp_review(review):
    return {
        "date": review["date"],
        "rating": review["rating"],
        "comment": review["comment"]["text"],
        "language": review["comment"]["language"],
        "details": review.get("feedback", pd.NA),
        "source": "Yelp",
    }


def process_google_review(review):
    return {
        "date": review["iso_date_of_last_edit"],
        "rating": review["rating"],
        "comment": review["snippet"],
        "language": "en",
        "details": review.get("details", pd.NA),
        "source": "Google",
    }


def get_combined_data(google_reviews, yelp_reviews):
    combined_data = []
    combined_data.extend(process_google_review(review) for review in google_reviews)
    combined_data.extend(process_yelp_review(review) for review in yelp_reviews)
    return pd.DataFrame(combined_data)
