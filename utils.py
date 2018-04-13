from scipy.spatial.distance import cosine

class utils():
    def loadVectors(self, fil):
        pass


class Vectors():
    def __init__(self):
        f = open('index.txt', 'rt')
        self.vecs = open('crawl-300d-2M.vec', 'rt')
        self.locations = {}
        for line in f:
            try:
                word, location = line.split()
            except:
                continue
            location = int(location)
            self.locations[word] = location

    def wordExists(self, word):
        if word in self.locations:
            return True
        else:
            return False

    def getVec(self, word):
        if word not in self.locations:
            return 'error'
        location = self.locations[word]
        self.vecs.seek(location)
        line = self.vecs.readline()
        line_split = line.split()
        return [float(i) for i in line_split[1:]]

if __name__ == '__main__':
    vec = Vectors()
