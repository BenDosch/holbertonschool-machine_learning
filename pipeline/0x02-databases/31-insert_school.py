"""Module that contains the function insert_school."""

from pymongo import MongoClient


def insert_school(mongo_collection, **kwargs):
    """Function that inserts a new document in a collection based on kwargs."""
    docs = mongo_collection.insert(kwargs)
    return docs


if __name__ == "__main__":
    list_all = __import__('30-all').list_all
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.my_db.school
    id = insert_school(collection, name="UCSF",
                       address="505 Parnassus Ave")
    print("New school created: {}".format(id))

    schools = list_all(collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'),
                                  school.get('name'),
                                  school.get('address', "")))
