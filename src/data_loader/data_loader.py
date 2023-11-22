import Pyro5.api
import mariadb
# from ..ML_Model import ModelStruct

# read permitted mac addresses


class ServerDataLoader(object):
    def __init__(self):
        self.modelInstances = {}
        self.dbConn = None
        self.connectDatabase()
        print("Initialized DataLoader")

    def connectDatabase(self):
        try:
            self.dbConn = mariadb.Connection(
                user="root",
                password="mariadb",
                host="localhost",
                port=3306,
                database="owrt_test"
                # pool_name="pool1"
            )
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")

    def getModelInstance(self, mac):
        if mac in self.modelInstances:
            return self.modelInstances[mac]
        else:
            return None  # should maybe initialize a new one? idk

    def loadModel(self, mac, path):
        self.modelInstances[mac] = ModelInstance()  # idk how to load the model lol
        print("Loading model")
        return self.modelInstances[mac]

    def getPackets(self, timerange, category, model):
        # conn = self.dbConn.get_connection()  # need to fix pooling (aka just allocate them)
        cursor = self.dbConn.cursor(dictionary=True)  # conn.cursor()
        cursor.execute(
            """SELECT packet.PACK_ID, packet.PACK_ORIGIN_IP, packet.PACK_DEST_IP, packet.PACK_PAYLOAD,
                        pack_label.PACK_CLASS, pack_label.PACK_CONFIDENCE
                FROM packet
                INNER JOIN pack_label ON pack_label.PACK_ID = packet.PACK_ID;"""
        )
        results = cursor.fetchall()
        print(results)
        # resultsdf = pd.DataFrame(results)
        # print(resultsdf)
        # if results is dict:
        return results  # df.to_dict("rows")
        # else:
        #    return pd.read_csv("../sample.csv").to_dict(orient="rows")
    # resultsdf.to_dict(orient="rows")  # pd.read_csv("../sample.csv").to_dict(orient="rows")


class ModelInstance(object):
    def __init__(self):
        print(":)")
        # self.model = ModelStruct.Conv1DClassifier()
        # ModelStruct.train_model(self.model)

    def feed(payload):
        # dataObject = self.model.generateDataObject(payload, otherthings)
        return None


loader = ServerDataLoader()


@Pyro5.api.expose
class DataLoaderInterface(object):
    def __init__(self):
        self.mac = None

    def connectModel(self, path):
        self.model = loader.getModelInstance(mac=self.mac)
        if self.model is None:
            self.model = loader.loadModel(mac=self.mac, path=path)

    def testPrint(self):
        loader.testPrint()

    def setMac(self, mac1):
        self.mac = mac1

    def getMac(self):
        return self.mac

    def getPackets(self, timerange, category):
        return loader.getPackets(timerange=timerange, category=category, model=self.model) 
        # return pd.read_csv("../sample.csv").to_dict(orient="rows")

    def uploadPackets(self, payload, mac):
        return None


daemon = Pyro5.api.Daemon(port=58116)
uri = daemon.register(DataLoaderInterface, objectId="dataloader")

print(uri)
daemon.requestLoop()
