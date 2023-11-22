import psycopg

class db_credentials:
	database:str
	user:str
	host:str
	password:str

	def __init__(self, dbname, user, host, pswd):
		self.dbname = dbname
		self.user = user
		self.host = host
		self.password = pswd

	def cred_dict(self):
		return vars(self)


def _connect_db(creds):
	db = None
	try:
		db = psycopg.connect(**creds.cred_dict())
	except (psycopg.DatabaseError) as e:
		print(e)
	return db

def _close_db(db):
	db.close()

def query_db(credentials, query):
	db = _connect_db(credentials)
	if (db == None):
		exit(-1)
	with db.cursor() as cur:
		try:
			cur.execute("SELECT version()")
			response = cur.fetchone()
			print(response)
		except (psycopg.DatabaseError) as e:
			print(e)
	_close_db(db)

if __name__ == "__main__":
	creds = db_credentials("senior", "root", "localhost", "joe")
	query_db(creds, None)
