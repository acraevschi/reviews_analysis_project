from serpapi import GoogleSearch
from config import API_KEY, RELEVANT_LANGS


def get_google_reviews(place_id, next_page_token, num="20"):
    params = {
        "engine": "google_maps_reviews",
        "place_id": place_id,
        "next_page_token": next_page_token,
        "num": num,
        "hl": "en",
        "api_key": API_KEY,
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results["reviews"]


def get_yelp_reviews(place_id):
    reviews = []
    for lang in RELEVANT_LANGS:
        params = {
            "engine": "yelp_reviews",
            "place_id": place_id,
            "api_key": API_KEY,
            "hl": lang,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        reviews.extend(results["reviews"])
    return reviews
