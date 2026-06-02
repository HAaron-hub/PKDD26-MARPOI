import argparse
import os
# Total tasks: NYC 1026 TKY 2264 CA 4189
case_num = 100

def readTrain(filePath, datasetName):
    longs = dict()
    pois = dict()
    with open(filePath, 'r') as file:
        lines = file.readlines()
    for line in lines[1:]:
        data = line.split(',')
        time, u, lati, longi, i, category, catID = data[1], data[5], data[6], data[7], data[8], data[10], data[9]
        if i not in pois:
            pois[i] = {"latitude": lati, "longitude": longi, "category": category, "categoryID": catID}
        if u not in longs:
            longs[u] = list()
        longs[u].append((i, time))
    return longs, pois

def readTest(filePath, datasetName):
    recents = dict()
    pois = dict()
    targets = dict()
    traj2u = dict()
    with open(filePath, 'r') as file:
        lines = file.readlines()
    for line in lines[1:]:
        data = line.split(',')
        time, trajectory, u, lati, longi, i, category, catID = data[1], data[3],data[5], data[6], data[7], data[8], data[10], data[9]
        if i not in pois:
            pois[i] = dict()
            pois[i]["latitude"] = lati
            pois[i]["longitude"] = longi
            pois[i]["category"] = category
            pois[i]["categoryID"] = catID
        if trajectory not in traj2u:
            traj2u[trajectory] = u
        if trajectory not in recents:
            recents[trajectory] = list()
            recents[trajectory].append((i, time))
        else:
            if trajectory in targets:
                recents[trajectory].append(targets[trajectory])
            targets[trajectory] = (i, time)
    return recents, pois, targets, traj2u

def getData(datasetName):
    if datasetName == 'nyc':
        filePath = './data/nyc/new_{}_sample.csv'
    elif datasetName == 'tky':
        filePath = './data/tky/new_{}_sample.csv'
    elif datasetName == 'ca':
        filePath = './data/ca/new_{}_sample.csv'
    else:
        raise NotImplementedError
    trainPath = filePath.format('train')
    testPath = filePath.format('test')

    longs, poiInfos = readTrain(trainPath, datasetName)
    recents, testPoi, targets, traj2u = readTest(testPath, datasetName)
    poiInfos.update(testPoi)
    targets = dict(list(targets.items())[:case_num])

    return longs, recents, targets, poiInfos, traj2u

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model to be run.', required=True)
    parser.add_argument('-d', '--datasetName', type=str, choices=['nyc', 'tky', 'ca'], default='nyc',help='nyc/tky/ca')
    args = parser.parse_args()

    data = getData(args.datasetName)
    path = './output/{}/{}'.format(args.model, args.datasetName)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.model == 'MARPOI':
        from models.MARPOI import MAPOI
        model = MAPOI()
    else:
        raise NotImplementedError

    results = model.run(data, args.datasetName, args.model)
    results = 'ACC@1: {:.4f}, ACC@5: {:.4f}, MRR@5: {:.4f}'.format(results[0], results[1], results[2])
   # Build output file path with extension.
    resultPath = './results/{}_{}.txt'.format(args.model, args.datasetName)

    # Ensure target directory exists.
    os.makedirs(os.path.dirname(resultPath), exist_ok=True)

    # Write result to file.
    with open(resultPath, 'w') as file:
        file.write(results)