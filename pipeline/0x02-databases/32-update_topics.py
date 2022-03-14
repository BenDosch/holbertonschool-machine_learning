"""Module that contains the function update_topics."""
from pymongo import MongoClient


def update_topics(mongo_collection, name, topics):
    """Function that changes all topics of a school document based on the
    name."""
    mongo_collection.update_many({'name': name},
                                 {'$set': {'name': name, 'topics': topics}})


if __name__ == "__main__":
    list_all = __import__('30-all').list_all
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school
    update_topics(school_collection, "Holberton school",
                  ["Sys admin", "AI", "Algorithm"])

    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'),
                                  school.get('name'),
                                  school.get('topics', "")))

    update_topics(school_collection, "Holberton school", ["iOS"])

    schools = list_all(school_collection)
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'),
                                  school.get('name'),
                                  school.get('topics', "")))
