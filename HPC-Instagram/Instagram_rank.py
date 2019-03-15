import json
import time
from mpi4py import MPI
import numpy

"""
Initialize the MPI communicator
comm_size is the number of the nodes.
comm_rank is the sequence number of the nodes and starts from 0.
"""
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()
# print(comm_size)
# print(comm_rank)

# Set the Start time
start_time = time.time()

"""
Set the relevant arrays
mel_grid is the array of the grids for the Melbourne, including A1,A2,B1,C1...etc.
coordinates is the array which contains the coordinates for each Instagram post. 
"""
mel_grid = []
# global coordinates
coordinates = []

"""
This function created a dictionary structure set called 'grids'. Then it reads and takes the 'id' (A1,A2,B1,C1...) 
and the coordinates of them in the melbGrid.json, and put them together in the array mel_grid. 
In addition, add the key 'row' and 'column'in 'grids' to count the total number of posts in row/column next 
through the slicing. 
Lastly, it also adds the key 'related_post_number' to ignore the posts outside the grids of Melbourne .
"""
def create_mel_grid(file):
    # f1 = open('dataset/melbGrid.json','r'):
    # f1.close()

    with open(file) as f1:
        features = json.load(f1)
        for each_row in features["features"]:
            # grids = {}
            # Created a dictionary structure set called 'grids'
            grids = dict()
            grids["id"] = each_row["properties"]["id"]
            grids["xmin"] = each_row["properties"]["xmin"]
            grids["xmax"] = each_row["properties"]["xmax"]
            grids["ymin"] = each_row["properties"]["ymin"]
            grids["ymax"] = each_row["properties"]["ymax"]
            # Take 'A' 'B' 'C' 'D'
            grids["row"] = each_row["properties"]["id"][:1]
            # Take '1' '2' '3' '4' '5'
            grids["column"] = each_row["properties"]["id"][1:2]
            grids["related_post_number"] = 0
            mel_grid.append(grids)
    return None

"""
This function reads file and takes the coordinates to save in the array 'coordinate'. 
Through the 'try' and 'except', it can ignore posts without coordinates.
By opening the json file, I found all the coordinate data are in ["rows"]["doc"], 
so just read it and save in single_coord and save all the single_coord in the array coordinates.
"""
def get_coordinate(file):
    # 1 node and 1 core
    coordinate = []
    if comm_size <= 1 and comm_rank == 0:
        with open(file) as f2:
            # This part which is with pound sign only works for tinyInstagram.json
            # When read medium/big files, json.load will show errors. So use json.loads instead.
            #
            # coords = json.load(f2)
            # for each_row in coords["rows"]:
            #     try:
            #         single_coord = dict()
            #         coord = each_row["doc"]["coordinates"]["coordinates"]
            #         # print(coord)
            #         single_coord["latitude"] = coord[0]
            #         single_coord["longitude"] = coord[1]
            #         # print(single_coord)
            #         coordinates.append(single_coord)
            #     except:
            #         continue
            #
            # json.loads:
            for each_row in f2:
                try:
                    # json.loads only convert 'string' type to 'dictionary' type
                    # Use slicing to remove the comma and line break to obtain the whole string in each row.
                    coord = json.loads(each_row[:-2])
                    # print(coord)
                    single_coord = dict()
                    single_coord["longitude"] = coord["doc"]["coordinates"]["coordinates"][1]
                    single_coord["latitude"] = coord["doc"]["coordinates"]["coordinates"][0]
                    # print(single_coord)
                    coordinate.append(single_coord)
                except:
                    try:
                        # read the last line (without comma)
                        coord = json.loads(each_row[:-1])
                        # print(coord)
                        single_coord = dict()
                        single_coord["longitude"] = coord["doc"]["coordinates"]["coordinates"][1]
                        single_coord["latitude"] = coord["doc"]["coordinates"]["coordinates"][0]
                        # print(single_coord)
                        coordinate.append(single_coord)
                    except:
                        continue
    # Other resources (1 node 4 cores/2 nodes 8 cores)
    # Since scatter only allow equally divide, so have to do split operation before do comm.scatter.
    elif comm_rank == 0:
        with open(file) as f3:
            for each_row in f3:
                try:
                    coord = json.loads(each_row[:-2])
                    # print(coord)
                    single_coord = dict()
                    single_coord["longitude"] = coord["doc"]["coordinates"]["coordinates"][1]
                    single_coord["latitude"] = coord["doc"]["coordinates"]["coordinates"][0]
                    # print(single_coord)
                    coordinate.append(single_coord)
                except:
                    try:
                        coord = json.loads(each_row[:-1])
                        # print(coord)
                        single_coord = dict()
                        single_coord["longitude"] = coord["doc"]["coordinates"]["coordinates"][1]
                        single_coord["latitude"] = coord["doc"]["coordinates"]["coordinates"][0]
                        # print(single_coord)
                        coordinate.append(single_coord)
                    except:
                        continue
        # coordinate = numpy.split(coordinate, comm_size, axis=0)
        coordinate = numpy.array_split(coordinate, comm_size)
        # print(coordinate)
    else:
        coordinate = None
    return coordinate

"""
This function ensure to only count posts in the grids of Melbourne. 
"""
def related_post_number(grids, longitude, latitude):
    for each_grid in grids:
        try:
            if (longitude >= each_grid["xmin"] and longitude <= each_grid["xmax"]) and (latitude >= each_grid["ymin"] and latitude <= each_grid["ymax"]):
                each_grid["related_post_number"] = each_grid["related_post_number"] + 1
        except:
            continue


# create_mel_grid('dataset/melbGrid.json')
create_mel_grid('melbGrid.json')
# print(mel_grid)
# coordinates = get_coordinate('dataset/tinyInstagram.json')
coordinates = get_coordinate('bigInstagram.json')
# coordinates = get_coordinate('mediumInstagram.json')
# print(coordinates)

"""
Here comes to the main function.
Select Instagram posts in the grids of Melbourne and save the result in the variable valid_number.
"""
if comm_size <= 1 and comm_rank == 0:
    for each_data in coordinates:
        related_post_number(mel_grid, each_data["longitude"], each_data["latitude"])
    valid_number = mel_grid
    # Delete unrelated data
    for each_data in valid_number:
        try:
            del each_data["xmax"]
            del each_data["xmin"]
            del each_data["ymax"]
            del each_data["ymin"]
        except:
            continue
    # print(valid_number)
else:
    # The master node distributes the tasks to different nodes by using comm.scatter
    # and gather the result through comm.gather.
    # comm.scatter only allow equally divide.
    coordinate = comm.scatter(coordinates, root=0)
    for each_data in coordinate:
        related_post_number(mel_grid, each_data["longitude"], each_data["latitude"])
    valid_number = comm.gather(mel_grid, root=0)
    # Shows typeerror below
    # for each_data in valid_number:
    #     try:
    #         del each_data["xmax"]
    #         del each_data["xmin"]
    #         del each_data["ymax"]
    #         del each_data["ymin"]
    #     except:
    #         continue
    # print(valid_number)

"""
Count the result and print
"""
if comm_rank == 0:
    row_number = {"A": 0, "B": 0, "C": 0, "D": 0}
    column_number = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    box_number = {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "B1": 0, "B2": 0, "B3": 0, "B4": 0, "C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0, "D3": 0, "D4": 0, "D5": 0}

    """
    Put the count result to the dictionaries above.
    """
    if comm_size <= 1:
        for each_data in valid_number:
            box_number[each_data["id"]] = each_data["related_post_number"] + box_number[each_data["id"]]
    else:
        # Operate on the valid_number[array["A1": 0, "A2": 0, ...], array["B1": 0, "B2": 0, ...], ...]
        for each_data in valid_number:
            for each_sub_data in each_data:
                box_number[each_sub_data["id"]] = each_sub_data["related_post_number"] + box_number[each_sub_data["id"]]

    # Count the result of row numbers
    for each_key in box_number:
        row_number[each_key[:1]] = box_number[each_key] + row_number[each_key[0:1]]

    # Count the result of column numbers
    for each_key in box_number:
        column_number[each_key[1:2]] = box_number[each_key] + column_number[each_key[1:2]]

    # Print all the result
    print("\nThe total number of posts in each grid box:\n")
    result = sorted(box_number, key=box_number.get, reverse=True)
    for each_number in result:
        print("%s has %d posts" % (each_number, box_number[each_number]))

    print("\nThe total number of posts in each row:\n")
    result = sorted(row_number, key=row_number.get, reverse=True)
    for each_number in result:
        print("Row %s has %d posts" % (each_number, row_number[each_number]))

    print("\nThe total number of posts in each column:\n")
    result = sorted(column_number, key=column_number.get, reverse=True)
    for each_number in result:
        print("Column %s has %s posts" % (each_number, column_number[each_number]))

    # Print the total time
    used_time = time.time() - start_time
    print("\nThe total time is: %.5f" % used_time)