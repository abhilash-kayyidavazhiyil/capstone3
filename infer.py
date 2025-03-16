import re
import ast
from neo4j import GraphDatabase
import cv2
import os

import spacy

def extract_triplets_and_objects(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    triplets = []
    objects = set()
    main_subject = "person"

    for token in doc:

        if token.dep_ in ["attr", "dobj"] and "wear" in token.head.lemma_:
            subject = main_subject  # Always use 'person'
            clothing_desc = (" ".join([child.text for child in token.lefts] + [token.text])).replace("a ","")
            triplets.append((subject, "wear", clothing_desc))
            objects.add(subject)
            objects.add(clothing_desc)

        # Detect 'carrying', 'using', and similar actions
        if token.dep_ == "dobj" and token.head.lemma_ in ["carry", "use", "hold"]:
            subject = main_subject  # Always use 'person'
            triplets.append((subject, token.head.lemma_, token.text))
            objects.add(subject)
            objects.add(token.text)

        # Handle conjunctions for combined clothing descriptions
        if token.dep_ == "conj" and token.head.dep_ in ["attr", "dobj"]:
            subject = main_subject
            clothing_desc = (" ".join([child.text for child in token.lefts] + [token.text])).replace("a ","")
            triplets.append((subject, "wear", clothing_desc))
            objects.add(subject)
            objects.add(clothing_desc)

    objects = list(set(objects))

    return triplets, list(objects)


class KnowledgeGraph:
    def __init__(self, uri, user, password):
        """
        Initialize the KnowledgeGraph with a connection to the Neo4j database.

        Args:
            uri (str): The URI of the Neo4j database.
            user (str): The username for the Neo4j database.
            password (str): The password for the Neo4j database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the connection to the Neo4j database."""
        self.driver.close()


    def find_video_and_timestamp(self, objects):
        """
        Find videos and timestamps associated with the given objects.

        Args:
            objects (list): A list of objects to search for.

        Returns:
            list: A list of results containing video names, timestamps, and relationships.
        """
      
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:Frame)-[:CONTAINS]->(o:Object)
                WHERE ANY(word IN $objects WHERE o.name =~ ('.*' + word + '.*'))
                WITH f, COLLECT(o.name) AS objectList
                RETURN DISTINCT f.video_name AS videoName, f.timestamp AS timestamp, objectList
            """, objects=objects)

            # Collect results
            results = []
            for record in result:
                if len(objects) > 1:
                    
                    if objects[0] in record["objectList"] and objects[1] in record["objectList"]:
                        
                        results.append({
                            "videoName": record["videoName"],
                            "timestamp": record["timestamp"],
                            "objects": record["objectList"],
                        })
                else:
                    results.append({
                            "videoName": record["videoName"],
                            "timestamp": record["timestamp"],
                            "objects": record["objectList"],
                        })
            return results


def process_text(input_text):

    triplets, objects = extract_triplets_and_objects(input_text)

    return objects


# Function to read and return the image at a specific timestamp
def get_frame_at_time(video_path, timestamp):
    v_name = video_path.replace(".mp4","")
    frame = f"static/images/{timestamp}_{v_name}.jpg"
    return frame


from flask import Flask, render_template, request, redirect, url_for, flash
import secrets

secret_key = secrets.token_hex(16)  # Generates a random 32-character hex string
app = Flask(__name__)
app.secret_key = secret_key # Required for flashing messages

# Database connection parameters
uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"
kg = KnowledgeGraph(uri, user, password)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form.get('caption')
        search_objects = process_text(input_text)
        if "person" in search_objects:
            search_objects.remove("person")
        results = kg.find_video_and_timestamp(search_objects)

        video_dict = {}
        for result in results:
            if True:
                video_name = result['videoName']
                timestamp = result['timestamp']
                if video_name not in video_dict:
                    video_dict[video_name] = []
                if timestamp not in video_dict[video_name]:
                    video_dict[video_name].append(float(timestamp))
                
        return render_template('index.html', video_dict=video_dict, input_text=input_text)

    return render_template('index.html', video_dict={}, input_text='')

@app.route('/show_frame', methods=['POST'])
def show_frame():
    selected_video = request.form.get('selected_video')
    selected_timestamp = float(request.form.get('selected_timestamp'))

    video_path = f"Video_datasets/{selected_video}"  # Adjust the path as necessary
    frame = get_frame_at_time(selected_video, selected_timestamp)

    # Save the frame to a static directory or process it as needed
    return render_template('show_frame.html', frame_path=frame, timestamp=selected_timestamp)
   

if __name__ == "__main__":
    app.run(debug=True,port=8011)