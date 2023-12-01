import Pyro5.api
import dbManager
import math
import parser as packetParse
import pandas as pd
import threading
import psycopg2.extras
import sys
import os
import math
import networkFeed
# from ..ML_Model import ModelStruct  # for autocomplete
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/ML_Model")
import ModelStruct


# read permitted mac addresses
# TODO: replace ALL SQL with SqlAlchemy (easier and safer queries)


class ServerDataLoader(object):
    PACKET_COLS = {"pack_id", "pack_origin_ip", "pack_dest_ip", "protocol", "length", "t_delta", "ttl", "srcport", "destport"}
    PACK_CLASS_COLS = {"pack_class", "pack_confidence"}
    SQL_OPERATORS = {"equals": "=", "notEqual": "!=", "notContains": "NOT LIKE", "contains": "LIKE", "greaterThan": ">", "greaterThanOrEqual": ">=", "lessThan": "<", "lessThanOrEqual": "<="}

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
        if mac not in self.modelInstances:
            self.modelInstances[mac] = ModelInstance()
            self.modelInstances[mac].loadModel()
        return self.modelInstances[mac]

    def filter2Where(column, filter, isNumber):
        # temporary
        if isNumber:
            return f"({column} {ServerDataLoader.SQL_OPERATORS[filter['type']]} {filter['filter']}) "
        if filter['type'] == 'contains' or filter['type'] == 'notContains':
            filter['filter'] = f"%%{filter['filter']}%%"
        return f"({column} {ServerDataLoader.SQL_OPERATORS[filter['type']]} '{filter['filter']}') "

    def SqlWhereFromFilters(filters):
        # temporary
        whereString = ""
        for index, column in enumerate(filters):
            schemaCol = None
            if column in ServerDataLoader.PACKET_COLS:
                schemaCol = f"packet.{column}"
            elif column in ServerDataLoader.PACK_CLASS_COLS:
                schemaCol = f"pack_label.{column}"
            else:
                print(f"Schema discrepancy: {column}")
                return ""

            if index == 0:
                whereString += "WHERE "
            else:
                whereString += " AND "

            isNumber = filters[column]['type'] == 'number'

            if 'operator' in filters[column]:
                op = filters[column]['operator'] + " "
                whereString += op.join([ServerDataLoader.filter2Where(column=schemaCol, filter=condition, isNumber=isNumber) for condition in filters[column]['conditions']])
            else:
                whereString += ServerDataLoader.filter2Where(column=schemaCol, filter=filters[column], isNumber=isNumber)

            print(whereString)

        return whereString

    def getClassBins(self, timerange, binct):
        results = None
        with self.dbConn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            try:  # probably need join for time range. consider merging with getPackets query?
                # TODO: use range for timestamp and amount of time
                cursor.execute("""SELECT date_bin('1 hours', TIMESTAMP '2023-11-27 20:20:20', packet.time) as bucket,
                                    pack_label.pack_class, count(*) as ct
                               FROM PACK_LABEL
                               LEFT JOIN packet ON packet.PACK_ID = pack_label.PACK_ID
                               GROUP BY pack_label.pack_class, bucket
                               ORDER BY bucket, pack_label.pack_class""")
                results = cursor.fetchall()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
            except Exception as e:
                print(f"General exception: {e}")
        print(results)
        return [dict(row) for row in results]

    def getClassCounts(self, timerange):
        results = None
        with self.dbConn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            try:  # probably need join for time range. consider merging with getPackets query?
                cursor.execute("""SELECT pack_class, count(*) as ct FROM PACK_LABEL
                               GROUP BY pack_class""")
                results = cursor.fetchall()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
            except Exception as e:
                print(f"General exception: {e}")
        print(results)
        return [dict(row) for row in results]

    def getPacket(self, id, model):
        result = None
        with self.dbConn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            try:
                cursor.execute("""SELECT * FROM PACKET
                               LEFT JOIN pack_label ON pack_label.PACK_ID = packet.PACK_ID
                               WHERE packet.PACK_ID = %s""", (id,))
                result = cursor.fetchone()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
            except Exception as e:
                print(f"General exception: {e}")
        return dict(result)

    def getPackets(self, timerange, category, requestData, model):
        if requestData is None:
            return []

        # TODO: replace all these unescaped SQL hacks with an actual SQL generator
        sqlParams = {
            'offset': requestData['startRow'],
            'limit': requestData['endRow'] - requestData['startRow']
        }
        if 'sortModel' not in requestData or len(requestData['sortModel']) == 0:
            sqlParams['sortCol'] = 'PACK_ID'
            sqlParams['sortDir'] = 'DESC'
        else:
            if requestData['sortModel'][0]['colId'] in ServerDataLoader.PACKET_COLS:
                sqlParams['sortCol'] = f"packet.{requestData['sortModel'][0]['colId'].upper()}"
            else:
                sqlParams['sortCol'] = f"pack_label.{requestData['sortModel'][0]['colId'].upper()}"

            sqlParams['sortDir'] = requestData['sortModel'][0]['sort'].upper()

        if len(requestData['filterModel']) == 0:
            sqlParams['where'] = ''
        else:
            sqlParams['where'] = ServerDataLoader.SqlWhereFromFilters(requestData['filterModel'])

        queryString = f"""SELECT packet.PACK_ID, packet.PACK_ORIGIN_IP, packet.PACK_DEST_IP, packet.SRCPORT, packet.DESTPORT,
                                pack_label.PACK_CLASS, pack_label.PACK_CONFIDENCE, packet.PROTOCOL, packet.LENGTH, packet.T_DELTA, packet.TTL, count(*) OVER() AS full_count
                        FROM packet
                        LEFT JOIN pack_label ON pack_label.PACK_ID = packet.PACK_ID
                        {sqlParams['where']}
                        ORDER BY {sqlParams['sortCol']} {sqlParams['sortDir']}
                        OFFSET %(offset)s
                        LIMIT %(limit)s;"""

        results = None
        with self.dbConn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            try:
                cursor.execute(queryString, sqlParams)
                results = cursor.fetchall()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
            except Exception as e:
                print(f"General exception: {e}")

        if isinstance(results, dict):
            return dict(results)
        if results is None:
            return []
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
                total_len = payload['total_len'][i]
                protocol = payload['protocol'][i]
                t_delta = payload['t_delta'][i]
                src = payload['src'][i]
                dest = payload['dest'][i]
                time = payload['time'][i]
                srcport = payload['srcport'][i]
                destport = payload['destport'][i]
                if srcport == "":
                    srcport = 'NULL'
                if destport == "":
                    destport = 'NULL'

                if i > 1:
                    values += ","

                payloadData = ",".join([str(payload[byteColumn][i]) for byteColumn in payloadColumns])

                values += f"(DEFAULT, {str(ttl)}, {str(total_len)}, '{str(protocol)}', {str(t_delta)}, '{str(payloadData)}', '{str(src)}', '{str(dest)}', to_timestamp({str(time)}), {srcport}, {destport})"

            try:
                results = cursor.execute(f"INSERT INTO packet (pack_id, ttl, length, protocol, t_delta, pack_payload, pack_origin_ip, pack_dest_ip, time, srcport, destport) VALUES {values} RETURNING pack_id;")
                self.dbConn.commit()
                results = cursor.fetchall()  # retrieve IDs from insert
                cursor.close()
            except Exception as e:
                print("General exception: {0}".format(e))
        if results is None:
            print("An attempt was made to insert packets into the database, but no row IDs were returned for some reason. There should be a SQL exception above.")
            return None
        return [x[0] for x in results]

    def insertLabels(self, ids, data, model_id):
        print(f"[model:{model_id}]: Inserting packet labels to database")
        id_pred_conf = zip(ids, data.predictions_string, data.prediction_confidence)

        with self.dbConn.cursor() as cursor:
            values = b','.join([cursor.mogrify("(%s, %s, %s, %s)", (str(model_id), str(x[0]), str(x[1]), str(x[2].item()))) for x in id_pred_conf])

            try:
                cursor.execute(f"INSERT INTO pack_label (MODEL_ID, PACK_ID, PACK_CLASS, PACK_CONFIDENCE) VALUES {values.decode()};")
                self.dbConn.commit()
            except psycopg2.errors.SyntaxError as e:
                print("Syntax error: {0}".format(e))
            except Exception as e:
                print("General exception: {0}".format(e))

    def newPackets(self, payload):  # client shouldn't wait for this
        if sum([model.loaded for _, model in self.modelInstances.items()]) == 0:
            print("No models are loaded, so payloads are being ignored.")
            return None
        print(f"Received payload of length {len(payload['ttl'])}")
        ids = self.insertPackets(payload=payload)
        payload = payload.drop(columns=['src', 'dest', 'time', 'srcport', 'destport'])
        # strip columns that can't be passed into the model
        for _, model in self.modelInstances.items():
            if not model.loaded:
                continue
            dataObj = model.feed(payload)
            self.insertLabels(ids=ids, data=dataObj, model_id=1)


class ModelInstance(object):
    def __init__(self):
        if True:
            self.loaded = False
        else:
            print("Model loading was intentionally bypassed")

    def loadModel(self, save_name=None):
        try:
            if save_name is None:
                print("A model save was not specified, so the model is being trained again.")
                self.model = ModelStruct.get_model(path="Saves/models/MVP_model.pth")  # Use debug = True to use unitTesting dataset
                self.loaded = True
        except Exception as e:
            print(f"Loading error: {e}")

    def feed(self, payload):
        return self.model.generateDataObject(payload)


loader = ServerDataLoader()


@Pyro5.api.expose
class DataLoaderInterface(object):
    def __init__(self):
        self.mac = None

    def connectModel(self):
        # The connection message is excessive
        # print(f"[{self.mac}]: Connecting model")
        ModelStruct.Config.parameters["Dataset"][0] = "UnitTesting"
        self.model = loader.getModelInstance(mac=self.mac)
        if self.model is None:
            print(f"[{self.mac}]: Loading model")
            self.model = loader.loadModel(mac=self.mac)

    def setMac(self, mac1):
        self.mac = mac1

    def getMac(self):
        return self.mac

    def getClassBins(self, timerange, binct):
        return loader.getClassBins(timerange=timerange, binct=binct)

    def getClassCounts(self, timerange):
        return loader.getClassCounts(timerange=timerange)

    def getPacket(self, id):
        return loader.getPacket(id=id, model=None)

    def getPackets(self, timerange, category, requestData):
        return loader.getPackets(timerange=timerange, category=category, requestData=requestData, model=self.model)

    def uploadPackets(self, payload):
        print(f"[{self.mac}]: Received payload for database")
        payloaddf = pd.DataFrame.from_dict(payload)
        loader.newPackets(payloaddf)


# Pyro5.config.COMPRESSION = True
# Pyro5.config.SERIALIZER = "marshal"
daemon = Pyro5.api.Daemon(port=58116)
uri = daemon.register(DataLoaderInterface, objectId="dataloader")


if __name__ == "__main__":
    try:
        feedThread = threading.Thread(target=networkFeed.feedNetworkThr, args=(any, loader,))
        feedThread.start()
        daemon.requestLoop()
    except KeyboardInterrupt:
        print("Ending networkFeed and ServerDataLoader")
        daemon.close()
        feedThread.join()
