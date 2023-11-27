import Pyro5.api
import dbManager
import math
import parser as packetParse
import pandas as pd
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
        self.dbConn.autocommit = True

    def getModelInstance(self, mac):
        if mac in self.modelInstances:
            return self.modelInstances[mac]
        else:
            return None  # should maybe initialize a new one? idk

    def loadModel(self, mac):
        print(f"[{mac}]: Loading model")
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
                        LEFT JOIN pack_label ON pack_label.PACK_ID = packet.PACK_ID;"""
                )
                results = cursor.fetchall()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
        return [dict(row) for row in results]  # conv to dicts bc psycopg2 returns in a format pyro5 won't serialize

    def insertPackets(self, payload):
        print("Inserting packets to database")
        results = None
        with self.dbConn.cursor() as cursor:
            # should be nicer to do something like this:
            # try:
            #     values = ",".join([str(cursor.mogrify("(DEFAULT, %s, %s, %s, %s)", (payload['ttl'][i], payload['total_len'][i], payload['protocol'][i].encode(), payload['t_delta'][i]))) for i in range(1, len(payload['ttl']))])
            # except Exception as e:
            #     print("Exception: " + e)

            # temporary, but it works
            values = ""
            payloadColumns = [f"payload_byte_{x+1}" for x in range(1500)]
            for i in range(1, len(payload['ttl'])):
                ttl = payload['ttl'][i]
                if isinstance(ttl, dict):
                    ttl = payload['ttl'][i]['fields'][0]['show']
                total_len = payload['total_len'][i]
                protocol = payload['protocol'][i]
                t_delta = payload['t_delta'][i]
                if isinstance(t_delta, dict):
                    t_delta = payload['t_delta'][i]['fields'][0]['show']
                if i > 1:
                    values += ","

                payloadData = ",".join([str(payload[byteColumn][i]) for byteColumn in payloadColumns])

                values += f"(DEFAULT, {str(ttl)}, {str(total_len)}, '{str(protocol)}', {str(t_delta)}, '{str(payloadData)}')"

            try:
                results = cursor.execute(f"INSERT INTO packet (pack_id, ttl, length, protocol, t_delta, pack_payload) VALUES {values} RETURNING pack_id;")
                self.dbConn.commit()
                results = cursor.fetchall()  # retrieve IDs from insert
                cursor.close()
            except Exception as e:
                print("General exception: {0}".format(e))
        if results is None:
            print("An attempt was made to insert packets into the database, but no row IDs were returned for some unknown reason")
            return None
        return [x[0] for x in results]

    def insertLabels(self, ids, data, model_id):
        print("Inserting packet labels to database")
        id_pred_conf = zip(ids, data.predictions, data.prediction_confidence)

        with self.dbConn.cursor() as cursor:
            values = b','.join([cursor.mogrify("(%s, %s, %s, %s)", (str(model_id), str(x[0]), str(x[1].item()), str(x[2].item()))) for x in id_pred_conf])
            try:
                cursor.execute(f"INSERT INTO pack_label (MODEL_ID, PACK_ID, PACK_CLASS, PACK_CONFIDENCE) VALUES {values.decode()};")
                self.dbConn.commit()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
            except Exception as e:
                print("General exception: {0}".format(e))

    def newPackets(self, payload):  # client shouldn't wait for this
        ids = self.insertPackets(payload=payload)
        for key, model in self.modelInstances.items():
            dataObj = model.feed(payload)
            self.insertLabels(ids=ids, data=dataObj, model_id=1)


class ModelInstance(object):
    def __init__(self):
        if True:
            self.loadModel()
        else:
            print("Model loading was intentionally bypassed")

    def loadModel(self):
        ModelStruct.Config.parameters["Dataset"][0] = "UnitTesting"
        self.model = ModelStruct.Conv1DClassifier()
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
        print("Received payload for database")
        payloaddf = pd.DataFrame.from_dict(payload)
        print(payloaddf)
        loader.newPackets(payloaddf)


# Pyro5.config.COMPRESSION = True
# Pyro5.config.SERIALIZER = "marshal"
daemon = Pyro5.api.Daemon(port=58116)
uri = daemon.register(DataLoaderInterface, objectId="dataloader")


if __name__ == "__main__":
    # loader.loadModel("idk man")
    # loader.newPackets(packetParse.pcap2df("../samplePackets.pcapng"))
    daemon.requestLoop()
    daemon.close()
