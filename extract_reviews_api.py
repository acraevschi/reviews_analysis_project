from serpapi import GoogleSearch
import pandas as pd

API_KEY = "b2ce1b7cbe8aca760f953666aa25580d6507f1b2790273f19cbd89e70707abf4"


def get_google_reviews():  # for now only for Tres Amigos, later provide place_id and next_page_token as arguments
    params = {
        "engine": "google_maps_reviews",
        "place_id": "ChIJaW0PioYKkEcRK4WtJ2890M8",
        "next_page_token": "CAESBkVnSUlDQQ==",  # also a question how to get it
        "num": "20",
        "hl": "en",
        "api_key": API_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    reviews = results["reviews"]
    return reviews


### YELP REVIEWS
def get_yelp_reviews():  # for now only for Tres Amigos, later provide place_id as argument
    relevant_langs = ["de", "en"]

    reviews = []
    for lang in relevant_langs:
        params = {
            "engine": "yelp_reviews",
            "place_id": "IRkB5uh1wQVWPlOlZYR3Jw",
            "api_key": API_KEY,
            "hl": lang,  # yelp reviews have to be searched by all languages, for now only German and English :(
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        reviews.extend(results["reviews"])
    return reviews


def process_yelp_review(review):
    if "feedback" not in review.keys():
        review["feedback"] = pd.NA
    return {
        "date": review["date"],
        "rating": review["rating"],
        "comment": review["comment"]["text"],
        "language": review["comment"]["language"],
        "details": review["feedback"],  # Convert dict to JSON string
        "source": "Yelp",
    }


def process_google_review(review):
    if "details" not in review.keys():
        review["details"] = pd.NA
    return {
        "date": review["iso_date_of_last_edit"],
        "rating": review["rating"],
        "comment": review["snippet"],
        "language": "en",
        "details": review["details"],
        "source": "Google",
    }


def get_data(include_google=True, include_yelp=True):
    combined_data = []

    if include_google:
        google_reviews = get_google_reviews()
        combined_data.extend(process_google_review(review) for review in google_reviews)

    if include_yelp:
        yelp_reviews = get_yelp_reviews()
        combined_data.extend(process_yelp_review(review) for review in yelp_reviews)

    return pd.DataFrame(combined_data)


def __main__():
    data = get_data()
