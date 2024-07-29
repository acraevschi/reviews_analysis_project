from api_client import get_google_reviews, get_yelp_reviews
from data_processor import get_combined_data


def get_data(include_google=True, include_yelp=True):
    google_reviews = (
        get_google_reviews("ChIJaW0PioYKkEcRK4WtJ2890M8", "CAESBkVnSUlDQQ==")
        if include_google
        else []
    )
    yelp_reviews = get_yelp_reviews("IRkB5uh1wQVWPlOlZYR3Jw") if include_yelp else []
    return get_combined_data(google_reviews, yelp_reviews)


if __name__ == "__main__":
    data = get_data()
