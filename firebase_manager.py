import pyrebase

class FirebaseMangager:
    def __init__(self):
        self.firebaseConfig = {

            "apiKey": "AIzaSyBD0wvIj8vfK8gy1vo9NOJqrcoe8UevQpU",
            "authDomain": "fir-demo-5eca5.firebaseapp.com",
            "databaseURL": "https://firebase-demo.firebaseio.com",
            "projectId": "fir-demo-5eca5",
            "storageBucket": "fir-demo-5eca5.appspot.com",
            "messagingSenderId": "79532353704",
            "appId": "1:79532353704:web:9da5e62f024e2abcb8fa9d",
            "serviceAccount": "serviceAccountKey.json"
        }
        self.firebase_storage = pyrebase.initialize_app(self.firebaseConfig)
        self.storage = self.firebase_storage.storage()

    def download(self, fielpath, giveAname):
        # self.storage.child(f'files/{name}.zip').download("F:\python projects\mec308","myimages.zip")
        self.storage.child(fielpath).download(path=fielpath, filename=f"{giveAname}.zip")

    def listOfFiles(self):
        files = self.storage.list_files()
        return files
