import os
import cv2
from neo4j import GraphDatabase
from llava import process_input
import re
import ast



class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_data(self):
        with self.driver.session() as session:
            # Delete all relationships first
            session.run("MATCH ()-[r]->() DELETE r")
            # Then delete all nodes
            session.run("MATCH (n) DELETE n")

    def create_frame(self, video_name, timestamp, objects, relationships):
        with self.driver.session() as session:
            # Create a frame node
            session.run("CREATE (f:Frame {video_name: $video_name, timestamp: $timestamp})",
                        video_name=video_name, timestamp=timestamp)

            # Create object nodes
            for obj in objects:
                session.run("MERGE (o:Object {name: $name})", name=obj)

            # Create relationships
            for rel in relationships:
                session.run("""
                    MATCH (a:Object {name: $start}), (b:Object {name: $end})
                    CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)
                """, start=rel[0], end=rel[1], type=rel[2])
      
            # Link frame to objects
            for obj in objects:
                session.run("""
                    MATCH (f:Frame {video_name: $video_name, timestamp: $timestamp}), (o:Object {name: $name})
                    CREATE (f)-[:CONTAINS]->(o)
                """, video_name=video_name, timestamp=timestamp, name=obj)


    def find_video_and_timestamp(self, objects):
        with self.driver.session() as session:
            result = session.run("""        
                    MATCH (f:Frame)-[:CONTAINS]->(o:Object)
                    WHERE o.name IN $objects
                    WITH f, COLLECT(o.name) AS objectList
                    MATCH (o1:Object)-[r:RELATIONSHIP]->(o2:Object)
                    WHERE o1.name IN $objects AND o2.name IN $objects
                    RETURN DISTINCT f.video_name AS videoName, f.timestamp AS timestamp, objectList, 
                    o1.name AS relationshipStart, o2.name AS relationshipEnd, r.type AS relationshipType
                """, objects=objects)

            # Collect results
            results = []
            for record in result:
                results.append({
                    "videoName": record["videoName"],
                    "timestamp": record["timestamp"],
                    "objects": record["objectList"],
                    "relationshipType": (record["relationshipStart"],record["relationshipType"],record["relationshipEnd"])
                })
            return results



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



def extract_objects_and_relationships(X):
    # Parse the string as Python code
    parsed_code = ast.parse(X, mode='exec')

    # Initialize variables for objects and relationships
    objects = []
    relationships = []

    # Extract assignments from the parsed code
    for node in parsed_code.body:
        if isinstance(node, ast.Assign):
            # Extract 'objects' list
            if node.targets[0].id == "objects":
                objects = ast.literal_eval(node.value)
            # Extract 'relationships' list of tuples
            elif node.targets[0].id == "relationships":
                relationships = ast.literal_eval(node.value)

    return objects, relationships
# Video Processing and Object Detection
def process_videos(folder_path, kg):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(folder_path, file_name)
            process_video(video_path, kg)

def process_video(video_path, kg):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract timestamp and keyframes (e.g., every 30th frame)
        frame_id += 1
        if frame_id % fps == 0:  # Keyframe logic
            try:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Seconds    
                cv2.imwrite("input.jpg",frame)
                qry = '''check the image and identify the following details:

If there are any persons in the image.
For each detected person, specify:
The color of their clothing (upper body and lower body separately if possible).
Any objects they are holding, such as a mobile phone, bottle, or other items.
If no persons are detected in the image, return an empty string.'''
                # Pass frame to object detection model
                objects = process_input("input.jpg",qry)
           
                objects = objects.split("assistant")[1].split("<|im_end|>")[0].strip()
                if objects!="":
                    if True:   
                      
                        try:
                            # sentence = "If a person is near the dog and a car is blocking the person."
                            triplets, objects = extract_triplets_and_objects(objects)

                            kg.create_frame(video_name, timestamp, objects, triplets)
                            v_name = video_name.replace(".mp4","")
                            cv2.imwrite(f"static/images/{timestamp}_{v_name}.jpg",frame)
                            # cv2.imwrite("input.jpg",frame)

                        except Exception as e:
                            print(e,"-------------------------------------")
                        
                                
            except:
                pass
            print(video_name)
                            
    cap.release()



# Main Function
if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    kg = KnowledgeGraph(uri, user, password)
    kg.clear_data()
    
    # # Process videos in a folder
    video_folder = "Video_datasets"
    process_videos(video_folder, kg)

    # Query the graph for an object and relationship
    search_objects = ["Person", "Bottle"]
    results = kg.find_video_and_timestamp(search_objects)

    for result in results:
        print(f"Video: {result['videoName']}, Timestamp: {result['timestamp']}, Objects: {result['objects']}, Relationship: {result['relationshipType']}")


    kg.close()


