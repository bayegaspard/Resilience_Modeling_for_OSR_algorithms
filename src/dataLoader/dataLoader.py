import Pyro5.api
import dbManager
import parser as packetParse
import psycopg2.extras
import sys
import os
# from ..ML_Model import ModelStruct  # for autocomplete
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/ML_Model")
import ModelStruct


# read permitted mac addresses


class ServerDataLoader(object):
    def __init__(self):
        self.modelInstances = {}
        self.dbConn = None
        self.connectDatabase()
        print("Initialized ServerDataLoader")

    def connectDatabase(self):
        creds = dbManager.db_credentials("owrt-dev2", "postgres", "127.0.0.1", "joe", 5433)
        self.dbConn = dbManager._connect_db(creds)

    def getModelInstance(self, mac):
        if mac in self.modelInstances:
            return self.modelInstances[mac]
        else:
            return None  # should maybe initialize a new one? idk

    def loadModel(self, mac):
        self.modelInstances[mac] = ModelInstance()
        return self.modelInstances[mac]

    def getPackets(self, timerange, category, model):
        results = None
        with self.dbConn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            try:
                cursor.execute(
                    """SELECT packet.PACK_ID, packet.PACK_ORIGIN_IP, packet.PACK_DEST_IP, packet.PACK_PAYLOAD,
                                pack_label.PACK_CLASS, pack_label.PACK_CONFIDENCE, packet.PROTOCOL, packet.LENGTH, packet.T_DELTA, packet.TTL
                        FROM packet
                        INNER JOIN pack_label ON pack_label.PACK_ID = packet.PACK_ID;"""
                )
                results = cursor.fetchall()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
        return [dict(row) for row in results]  # conv to dicts bc psycopg2 returns in a format pyro5 won't serialize

    def insertPackets(self, payload):
        print("Inserting packets to database")
        # join all rows in a string (internet suggests executemany is worse)
        # get cursor
        # execute insert query that returns ids
        # return ids

    def insertLabels(self, ids, data):
        print("Inserting packet labels to database")
        # join all rows in a string (internet suggests executemany is worse)
        # get cursor
        # execute insert query

    def newPackets(self, payload):
        ids = self.insertPackets(payload=payload)
        for _, model in self.modelInstances:
            self.insertLabels(ids=ids, data=model.feed(payload))


class ModelInstance(object):
    def __init__(self):
        if False:
            print("Model loaded")
            self.loadModel()
        else:
            print("Model loading was intentionally bypassed")

    def loadModel(self):
        ModelStruct.Config.parameters["Dataset"][0] = "UnitTesting"
        self.model = ModelStruct.Conv1DClassifier()
        self.model.st
        ModelStruct.train_model(self.model)

    def feed(self, payload):
        return self.model.generateDataObject(payload)


loader = ServerDataLoader()


@Pyro5.api.expose
class DataLoaderInterface(object):
    def __init__(self):
        self.mac = None

    def connectModel(self):
        print(f"[{self.mac}]: Connecting model")
        ModelStruct.Config.parameters["Dataset"][0] = "UnitTesting"
        self.model = loader.getModelInstance(mac=self.mac)
        if self.model is None:
            print(f"[{self.mac}]: Loading model")
            self.model = loader.loadModel(mac=self.mac)

    def setMac(self, mac1):
        self.mac = mac1

    def getMac(self):
        return self.mac

    def getPackets(self, timerange, category):
        return loader.getPackets(timerange=timerange, category=category, model=self.model)

    def uploadPackets(self, payload):
        loader.insertPackets(payload)


daemon = Pyro5.api.Daemon(port=58116)
uri = daemon.register(DataLoaderInterface, objectId="dataloader")


if __name__ == "main":
    daemon.requestLoop()
    daemon.close()
