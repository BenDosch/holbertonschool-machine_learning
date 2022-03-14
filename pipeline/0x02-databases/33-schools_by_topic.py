"""Module that contains the function schools_by_topic."""

from pymongo import MongoClient


def schools_by_topic(mongo_collection, topic):
    """Function that returns the list of school having a specific topic."""
    return mongo_collection.find({'topics': topic})


if __name__ == "__main__":
    list_all = __import__('30-all').list_all
    insert_school = __import__('31-insert_school').insert_school
    schools_by_topic = __import__('33-schools_by_topic').schools_by_topic
    client = MongoClient('mongodb://127.0.0.1:27017')
    school_collection = client.my_db.school

    j_schools = [
        { 'name': "Holberton school", 'topics': ["Algo", "C",
                                                 "Python", "React"]},
        { 'name': "UCSF", 'topics': ["Algo", "MongoDB"]},
        { 'name': "UCLA", 'topics': ["C", "Python"]},
        { 'name': "UCSD", 'topics': ["Cassandra"]},
        { 'name': "Stanford", 'topics': ["C", "React", "Javascript"]}
    ]
    for j_school in j_schools:
        insert_school(school_collection, **j_school)

    schools = schools_by_topic(school_collection, "Python")
    for school in schools:
        print("[{}] {} {}".format(school.get('_id'),
                                  school.get('name'),
                                  school.get('topics', "")))
