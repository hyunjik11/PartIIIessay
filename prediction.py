import numpy
import scipy.linalg
import math
import struct
import os
import mmap
import SVD
import re
import csv
import pickle
import operator

# loads averages from filename. n is the number of movies/users
def loadAverages(filename, n):
	averages = []
	avgFile = open(filename,'rb')
	for m in xrange(n):
		str = avgFile.read(struct.calcsize('d'))
		avg = struct.unpack('d',str)
		averages.append(avg[0])
	avgFile.close()

	return averages

# mmap data file in mode 'MovieUser' or 'UserMovie'
def mmapData( filename, mode = None, addresses = None ):
	if mode == None:
		mode = 'MovieUser'
		
	if addresses == None:
		addresses = 'Full'
		
	if mode not in ('MovieUser', 'UserMovie', 'ResidualMovieUser', 'ResidualUserMovie'):
		print "Error: mode " + mode + " is undefined"

	if addresses not in ('Full', 'StartOnly'):
		print "Error: address " + addresses + " is undefined"
		
	ratingFile = open( filename, "rb" )
	
	if mode == 'ResidualMovieUser':
		itemSize = struct.calcsize('>H')
		str = ratingFile.read(struct.calcsize('>H'))
		numItems = struct.unpack('>H',str)[0]
	
		str = ratingFile.read(struct.calcsize('>I'))
		totalEntries = struct.unpack('>I',str)[0]

		entryLengthSize = struct.calcsize('>I')
		entrySize = struct.calcsize('>Id')
		
		print "numMovies:" + repr(numItems) + "\tNumber of entries:" + repr(totalEntries)
	elif mode == 'MovieUser':
		itemSize = struct.calcsize('>H')
		str = ratingFile.read(struct.calcsize('>H'))
		numItems = struct.unpack('>H',str)[0]
	
		str = ratingFile.read(struct.calcsize('>I'))
		totalEntries = struct.unpack('>I',str)[0]

		entryLengthSize = struct.calcsize('>I')
		entrySize = struct.calcsize('>IB')
		
		print "numMovies:" + repr(numItems) + "\tNumber of entries:" + repr(totalEntries)

	elif mode == 'UserMovie':
		itemSize = struct.calcsize('>I')
		str = ratingFile.read(struct.calcsize('>I'))
		numItems = struct.unpack('>I',str)[0]
		
		str = ratingFile.read(struct.calcsize('>I'))
		totalEntries = struct.unpack('>I',str)[0]
		
		entryLengthSize = struct.calcsize('>H')
		entrySize = struct.calcsize('>HB')

		print "numUsers:" + repr(numItems) + "\tNumber of entries:" + repr(totalEntries)
	elif mode == 'ResidualUserMovie':
		itemSize = struct.calcsize('>I')
		str = ratingFile.read(struct.calcsize('>I'))
		numItems = struct.unpack('>I',str)[0]
		
		str = ratingFile.read(struct.calcsize('>I'))
		totalEntries = struct.unpack('>I',str)[0]
		
		entryLengthSize = struct.calcsize('>H')
		entrySize = struct.calcsize('>Id')

		print "numUsers:" + repr(numItems) + "\tNumber of entries:" + repr(totalEntries)
	
	ratingFile.close()
	
	ratingFile = os.open( filename, os.O_RDONLY )
	
	totalSize = itemSize +  struct.calcsize('>I') + numItems * entryLengthSize + totalEntries * entrySize
	
	filemap = mmap.mmap( ratingFile, totalSize, mmap.MAP_SHARED, mmap.PROT_READ )
	
	if mode == 'ResidualMovieUser':
		address = 0
		
		str = filemap.read(struct.calcsize('>H'))
		address = address + struct.calcsize('>H')
	
		str = filemap.read(struct.calcsize('>I'))
		address = address + struct.calcsize('>I')		

		if addresses == 'StartOnly':
			startAddress = address

		if addresses == 'Full':
			startAddress = []
			
			for m in xrange(numItems):
	
				startAddress.append(address)
	
				# read in number of user ratings
				str = filemap.read(struct.calcsize('>I'))
				length = struct.unpack('>I',str)[0]
				address = address + struct.calcsize('>I') + length * struct.calcsize('>Id')
				
				filemap.seek( length * struct.calcsize('>Id'), 1 )
				
	elif mode == 'MovieUser':
		address = 0
		
		str = filemap.read(struct.calcsize('>H'))
		address = address + struct.calcsize('>H')
	
		str = filemap.read(struct.calcsize('>I'))
		address = address + struct.calcsize('>I')		

		if addresses == 'StartOnly':
			startAddress = address

		if addresses == 'Full':
			startAddress = []
			
			for m in xrange(numItems):
	
				startAddress.append(address)
	
				# read in number of user ratings
				str = filemap.read(struct.calcsize('>I'))
				length = struct.unpack('>I',str)[0]
				address = address + struct.calcsize('>I') + length * struct.calcsize('>IB')
				
				filemap.seek( length * struct.calcsize('>IB'), 1 )
				
	elif mode == 'UserMovie':
		address = 0
		
		str = filemap.read(struct.calcsize('>I'))
		address = address + struct.calcsize('>I')
	
		str = filemap.read(struct.calcsize('>I'))
		address = address + struct.calcsize('>I')		

		if addresses == "StartOnly":
			startAddress = address

		if addresses == "Full":
			startAddress = []
			
			for m in xrange(numItems):
	
				startAddress.append(address)
	
				# read in number of movie ratings
				str = filemap.read(struct.calcsize('>H'))
				length = struct.unpack('>H',str)[0]
				address = address + struct.calcsize('>H') + length * struct.calcsize('>HB')
				
				filemap.seek( length * struct.calcsize('>HB'), 1 )
	elif mode == 'ResidualUserMovie':
		address = 0
		
		str = filemap.read(struct.calcsize('>I'))
		address = address + struct.calcsize('>I')
	
		str = filemap.read(struct.calcsize('>I'))
		address = address + struct.calcsize('>I')		

		if addresses == 'StartOnly':
			startAddress = address

		if addresses == 'Full':
			startAddress = []
			
			for m in xrange(numItems):
	
				startAddress.append(address)
	
				# read in number of user ratings
				str = filemap.read(struct.calcsize('>H'))
				length = struct.unpack('>H',str)[0]
				address = address + struct.calcsize('>H') + length * struct.calcsize('>Id')
				
				filemap.seek( length * struct.calcsize('>Id'), 1 )
				
				
	return filemap, startAddress

# load data file in mode 'MovieUser' or 'UserMovie'
def loadData( filename, mode = None ):
	if mode == None:
		mode = 'MovieUser'
	
	if mode not in ('MovieUser', 'UserMovie'):
		print "Error: mode " + mode + " is undefined"
	
	print "Loading Data from " + filename

	ratingFile = open( filename, "rb" )
	
	if mode == 'MovieUser':
		str = ratingFile.read(struct.calcsize('>H'))
		numMovies = struct.unpack('>H',str)[0]
	
		str = ratingFile.read(struct.calcsize('>I'))
		totalEntries = struct.unpack('>I',str)[0]

		print "numMovies:" + repr(numMovies) + "\tNumber of entries:" + repr(totalEntries)

		M = [ {} for m in range(numMovies) ]
	
		for m in xrange(numMovies):
			# read in number of user ratings
			str = ratingFile.read(struct.calcsize('>I'))
			length = struct.unpack('>I',str)[0]

			for l in xrange(length):
				str = ratingFile.read(struct.calcsize('>IB'))
				R = struct.unpack('>IB',str)

				M[m][R[0]] = R[1]

		print "Finished loading " + filename

	return M

# cosine similarity
def cosineSim( A, B ):
	return (A * B.T)[0,0]/(scipy.linalg.norm(A) * scipy.linalg.norm(B))

# cosine similarity
def cosineSimNorms( A, B, normA, normB ):
	return (A * B.T)[0,0]/(normA * normB)

# returns a list of items with similarity > simThreshold and sim < antiThreshold
def findSimilarAntiList( index, list, M, simThreshold, antiThreshold, norms ):
	similar = []
	anti = []
	if numpy.isnan(norms[index]): norms[index] = scipy.linalg.norm(M[index,:])
	for i in list:
		if i == index: continue
		if numpy.isnan(norms[i]): norms[i] = scipy.linalg.norm(M[i,:])
		sim = cosineSimNorms( M[index,:], M[i,:], norms[index], norms[i] )
		if sim > simThreshold:
			similar.append((i,sim))
		if sim < antiThreshold:
			anti.append((i,sim))

	return similar, anti

# returns a list of items with similarity > simThreshold and sim < antiThreshold
def findSimilarAnti( index, M, simThreshold, antiThreshold, norms ):
	similar = []
	anti = []
	if numpy.isnan(norms[index]): norms[index] = scipy.linalg.norm(M[index,:])
	dot = M[index,:] * M.T
	for i in xrange(M.shape[0]):
		if i == index: continue
		if numpy.isnan(norms[i]): norms[i] = scipy.linalg.norm(M[i,:])
		sim = dot[0,i] / ( norms[index] * norms[i] )
		if sim > simThreshold:
			similar.append((i,sim))
		if sim < antiThreshold:
			anti.append((i,sim))

	return similar, anti

# returns a list of items with similarity > threshold
def findSimilar( index, M, threshold ):
	similar = []
	for i in xrange(M.shape[0]):
		if i == index: continue
		sim = cosineSim( M[index,:], M[i,:] )
		if sim > threshold:
			similar.append((i,sim))
	return similar

# returns a list of items with similarity < threshold
def findAnti( index, M, threshold ):
	anti = []
	for i in xrange(M.shape[0]):
		if i == index: continue
		sim = cosineSim( M[index,:], M[i,:] )
		if sim < threshold:
			anti.append((i,sim))		
	return anti

def binarySearchRating( movie, userFile, startAddress, low, high ):
	length = high - low + 1

	if length < 10:
		userFile.seek( startAddress + low * struct.calcsize('HB') )
		for l in xrange(length):
			str = userFile.read(struct.calcsize('HB'))
			R = struct.unpack('HB',str)

			# HACK! Assumes ordering!!
			# FIXME
			movieID = R[0]-1
			if movieID > movie: break

			if int(movieID) == int(movie):
				return int(R[1])
	else:
		index = int( (high-low)/ 2.0 ) + low
		userFile.seek( startAddress + index * struct.calcsize('HB') )

		str = userFile.read(struct.calcsize('HB'))
		R = struct.unpack('HB',str)

		movieID = R[0]-1
	
		if int(movieID) == int(movie):
			return int(R[1])

		if int(movieID) > int(movie):
			return binarySearchRating( movie, userFile, startAddress, low, index - 1 )
		else:
			return binarySearchRating( movie, userFile, startAddress, index + 1, high )

	return -1

def findRating( user, movie, userFile, userAddress ):
	userFile.seek( userAddress[user] )
	str = userFile.read(struct.calcsize('H'))
	length = struct.unpack('H',str)[0]

	return binarySearchRating( movie, userFile, userAddress[user] + struct.calcsize('H'), 0, length-1 )

def getRating( user, movie, mapFile, address, mode = None ):
	if mode == None:
		mode = 'MovieUser'
		
	if mode not in ('MovieUser', 'UserMovie'):
		print "Error: mode " + mode + " is undefined"

	if mode == 'MovieUser':
		mapFile.seek( address[movie] )
		str = mapFile.read(struct.calcsize('>I'))
		length = struct.unpack('>I',str)[0]

		for l in xrange(length):
			str = mapFile.read(struct.calcsize('>IB'))
			R = struct.unpack('>IB',str)

			if int(R[0]) == int(user):
				return int(R[1])
	elif mode == 'UserMovie':
		mapFile.seek( address[user] )
		str = mapFile.read(struct.calcsize('>H'))
		length = struct.unpack('>H',str)[0]

		for l in xrange(length):
			str = mapFile.read(struct.calcsize('>HB'))
			R = struct.unpack('>HB',str)

			if int(R[0]) == int(movie):
				return int(R[1])
		
	return -1
		
def meanMovies( user, items, userFile, userAddress, offset ):
	total = 0.0
	count = 0.0
	for item in items:
		rating = findRating( user, item[0], userFile, userAddress )
		if rating != -1:
			count = count + item[1]
			total = total + (float(rating) - offset) * item[1]

	if count == 0.0: return 0.0
	return total / float(count)

def meanUsers( movie, items, userFile, userAddress, offset ):
	total = 0.0
	count = 0.0
	for item in items:
		rating = findRating( item[0], movie, userFile, userAddress )
		if rating != -1:
			count = count + item[1]
			total = total + (float(rating) - offset) * item[1]

	if count == 0.0: return 0.0
	return total / float(count)

def meanMoviesSVD( user, items, U, S, V, offset ):
	total = 0.0
	count = 0.0
	for item in items:
		p = U[user,:] * numpy.mat(numpy.diag(S)) * V[item[0],:].T
		count = count + item[1]
		total = total + (p[0,0] - offset) * item[1]

	if count == 0.0: return 0.0
	return total / float(count)
		
def meanUsersSVD( movie, items, U, S, V, offset ):
	total = 0.0
	count = 0.0
	for item in items:
		p = U[item[0],:] * numpy.mat(numpy.diag(S)) * V[movie,:].T
		count = count + item[1]
		total = total + (p[0,0] - offset) * item[1]

	if count == 0.0: return 0.0
	return total / float(count)

def getMovies( user, userFile, startAddress ):
	userFile.seek( startAddress )
	str = userFile.read(struct.calcsize('>H'))
	length = struct.unpack('>H',str)[0]
	
	movies = []

	for l in xrange(length):
		str = userFile.read(struct.calcsize('>HB'))
		R = struct.unpack('>HB', str)

		# FIXED?
		## FIXME: NEW userMovie.dat needed!
		if R[0] == 17770:
			print "ERROR in userMovie.dat!"
			
		movies.append(R[0])
	
	return movies

def getUsers( movie, movieFile, startAddress ):
	movieFile.seek( startAddress )
	str = movieFile.read(struct.calcsize('>I'))
	length = struct.unpack('>I',str)[0]

	users = []

	for l in xrange(length):
		str = movieFile.read(struct.calcsize('>IB'))
		R = struct.unpack('>IB', str)

		users.append(R[0])

	return users

def getMovieRatings( userFile, startAddress ):
	userFile.seek( startAddress )
	str = userFile.read(struct.calcsize('>H'))
	length = struct.unpack('>H',str)[0]
	
	#movies = []
	movies = numpy.zeros( (length,), numpy.int16 )
	ratings = numpy.zeros( (length,) )

	for l in xrange(length):
		str = userFile.read(struct.calcsize('>HB'))
		R = struct.unpack('>HB', str)

		# FIXED?
		## FIXME: NEW userMovie.dat needed!
		if R[0] == 17770:
			print "ERROR in userMovie.dat!"
			
		#movies.append( ( int(R[0]), int(R[1]) ) )
		movies[l] = R[0]
		ratings[l] = float(R[1])
		
	return movies, ratings
	
def getUserRatings( movieFile, startAddress ):
	movieFile.seek( startAddress )
	str = movieFile.read(struct.calcsize('>I'))
	length = struct.unpack('>I',str)[0]

	#users = []
	users = numpy.zeros( (length,), numpy.int )
	ratings = numpy.zeros( (length,) )

	for l in xrange(length):
		str = movieFile.read(struct.calcsize('>IB'))
		R = struct.unpack('>IB', str)

		#users.append( ( int(R[0]), int(R[1]) ) )
		users[l] = R[0]
		ratings[l] = float(R[1])
		
	return users, ratings

def predictBySVDNeighbours( user, movie, movieFile, movieAddress, userFile, userAddress, U, S, V, normMovies, normUsers ):
	simThreshold = 0.8
	antiThreshold = -1.0
	userMovies = getMovies( user, userFile, userAddress[user]  )
	similarMovies, antiMovies = findSimilarAntiList( movie, userMovies, V, simThreshold, antiThreshold, normMovies )
	moviesMean = meanMovies( user, similarMovies, userFile, userAddress, 0 )
	#features.append(meanMovies( user, antiMovies, userFile, userAddress, movieAverages[movie] ))
	
	simThreshold = 0.8
	antiThreshold = -1.0
	movieUsers = getUsers( movie, movieFile, movieAddress[movie] )
	similarUsers, antiUsers = findSimilarAntiList( user, movieUsers, U, simThreshold, antiThreshold, normUsers )
	usersMean = meanUsers( movie, similarUsers, userFile, userAddress, 0 )
	#features.append(meanUsers( movie, antiUsers, userFile, userAddress, userAverages[user] ))

	if moviesMean > 0.0 and usersMean > 0.0: return (moviesMean + usersMean) / 2.0
	if moviesMean > 0.0: return moviesMean
	if usersMean > 0.0: return usersMean

	return (U[user,:]*numpy.mat(numpy.diag(S))*V[movie,:].T)[0,0]


def features( user, movie, movieFile, movieAddress, userFile, userAddress, U, S, V, normMovies, normUsers, movieAverages, userAverages, dayOfWeek, week, year, holiday ):
	features = []

	features.append( movieAverages[ movie ] )
	features.append( userAverages[ user ] )

	feature_dayOfWeek = [ 0.0 for d in range(7) ]
	feature_dayOfWeek[dayOfWeek] = 1.0
	for d in xrange(7): features.append(feature_dayOfWeek[d])

	feature_week = [ 0.0 for w in range(54) ]
	feature_week[week] = 1.0
	for w in xrange(54): features.append(feature_week[w])

	feature_year = [ 0.0 for y in range(8) ]
	feature_year[year] = 1.0
	for y in xrange(8): features.append(feature_year[y])

	features.append(float(holiday))

	# SVD prediction
	p = (U[user,:] * numpy.mat(numpy.diag(S)) * V[movie,:].T)[0,0]

	p = max( p, 1.0 )
	p = min( p, 5.0 )

	features.append(p)

	return features

	movieFeatureSpace = V #(S*V.T).T
	userFeatureSpace = U #U * S

	simThreshold = 0.8
	antiThreshold = -0.8

	userMovies = getMovies( user, userFile, userAddress[user] )
	similarMovies, antiMovies = findSimilarAntiList( movie, userMovies, movieFeatureSpace, simThreshold, antiThreshold, normMovies )
	features.append(meanMovies( user, similarMovies, userFile, userAddress, movieAverages[movie] ))
	features.append(meanMovies( user, antiMovies, userFile, userAddress, movieAverages[movie] ))
	
	#simThreshold = 0.8
	#antiThreshold = -0.8
	#movieUsers = getUsers( movie, movieFile, movieAddress[movie] )
	#similarUsers, antiUsers = findSimilarAnti( user, userFeatureSpace, simThreshold, antiThreshold, normUsers )
	#features.append(meanUsers( movie, similarUsers, userFile, userAddress, userAverages[user] ))
	#features.append(meanUsers( movie, antiUsers, userFile, userAddress, userAverages[user] ))

	return features

def loadMovieAverages(filename = 'movieAveragesNoProbe.dat'):
	movieAverages = []
	movieAvgFile = open(filename,'rb')
	for m in xrange(17770):
		str = movieAvgFile.read(struct.calcsize('>d'))
		avg = struct.unpack('>d',str)
		movieAverages.append(avg[0])
	movieAvgFile.close()

	return movieAverages

def predictNormalizedSVD( user, movie, U, S, V, movieAverages ):
	pred = (U[u,:] * numpy.mat(numpy.diag(S)) * V[m,:].T)[0,0] + movieAverages[ movie ]

	pred = max( pred, 1.0 )
	pred = min( pred, 5.0 )
	
	return pred
	
def loadVB( VBFilename ):
	VBFile = open( VBFilename, "rb" )
	str = VBFile.read( struct.calcsize('>III') )
	r = struct.unpack('>III',str)
	numUsers = r[0]
	numMovies = r[1]
	rank = r[2]
	#rho = numpy.ones((rank,)) * 1/float(rank)
	str = VBFile.read( struct.calcsize('>d') )
	#tau = struct.unpack('d',str)[0]
	#sigma = numpy.zeros((rank,))
	#for r in xrange(rank):
	str = VBFile.read( struct.calcsize('>'+'d' * rank) )
		#sigma[r] = struct.unpack('d',str)[0]

	PU = numpy.mat(numpy.zeros((numUsers,rank)))
	PV = numpy.mat(numpy.zeros((numMovies,rank)))
	#covPV = numpy.zeros((numMovies,rank,rank))
	for u in xrange(numUsers):
		str = VBFile.read( struct.calcsize('>'+'d' * rank) )
		data = struct.unpack('>'+'d'*rank,str)
		for r in xrange(rank):
			PU[u,r] = data[r]
			
	for m in xrange(numMovies):
		str = VBFile.read( struct.calcsize('>'+'d' * rank) )
		data = struct.unpack('>'+'d'*rank,str)
		for r in xrange(rank):
			PV[m,r] = data[r]
	VBFile.close()
	
	return PU, PV

class Prediction:
	def __init__(self):
		self.SVD = []
		self.movieAverages = []
		self.titles = []
		self.norms = []
		self.userDict = []
		self.mmapFile = []
		self.refineSVD = None
		self.intercept = 0.0
		self.weights = numpy.array([  1.10552043e+00,   1.06280562e+00,   1.16521195e+00,   1.15557848e+00, 1.14855813e+00,   1.24595619e+00,   1.24694138e+00,   1.20906510e+00, 1.28430852e+00,   1.27358478e+00,   1.35120293e+00,   1.47781860e+00, 1.34472806e+00,   1.46486818e+00,   1.28884820e+00,   1.40381748e+00, 1.42082280e+00,   1.43480485e+00,   1.69116145e+00,   1.56163830e+00, 1.46557682e+00,  1.52025822e+00,   1.61233520e+00,   1.45983051e+00, 1.65578358e+00,   1.32226923e+00,   1.69972770e+00,   1.84679831e+00, 1.69446285e+00,   1.46407733e+00,   1.89950253e+00,   1.86899257e+00, 1.84986662e+00,   1.94316118e+00,   1.60390989e+00,   1.75202649e+00, 1.76281290e+00,   2.00891981e+00,   1.95296822e+00,   2.19943889e+00, 2.03976220e+00,   2.08147846e+00,   2.00504910e+00,   2.09129370e+00, 1.80114910e+00,   1.96831021e+00,   2.16509513e+00,   2.14036753e+00, 1.27838317e+00,   2.20098761e+00,  -3.09331588e-05,  -3.02321588e+02, 2.70448530e-05,  -1.13985603e+01,  7.16042656e-05,  -9.99935172e+00])
		self.W = None
		self.Z = None
		self.imdbFeatures = None
 		self.dayOfWeekOffset = 0
		self.weekOffset = 7
		self.yearOffset = 61
		self.holidayOffset = 68
  
		self.__doc__ = """
loadData( SVDFile, movieAveragesFile = 'movieAverages.dat', ratingFile = 'movieUser.dat' )

getMovieInfo( movie )

findMovieID( title )

getRating( user, movie )

findSimilarAnti( movie, simThreshold, antiThreshold )

printSimilarMovies( movie, simThreshold = 0.95, antiThreshold = -0.75 )

predict( user, movie )
		"""

	def loadData( self, SVDFile, movieAveragesFile = 'movieAverages.dat', ratingFile = 'movieUser.dat' ):
		numMovies = 17770
		numUsers = 480189
		
		U, S, V = SVD.loadSVD( SVDFile )
		
		S = numpy.mat(numpy.diag(S))
		self.SVD = (U,S,V)
		
		self.movieAverages = loadMovieAverages( movieAveragesFile )
	
		large = 1e300
		inf = large * large
		nan = inf - inf
	
		self.norms = numpy.ones((numMovies,)) * nan
		
		input = open('movie_titles.txt', 'rb')
		reader = csv.reader(input)
		self.titles = []
		for line in reader:
			self.titles.append(line[2])				
		input.close()
		
		pkl_file = open('usersDict.pkl', 'rb')	
		self.userDict = pickle.load(pkl_file)
		pkl_file.close()
		
		rFile, startAddress = mmapData( ratingFile )	
		self.mmapFile = ( rFile, startAddress )

		print "Done loading"
		
	def loadRefineSVD( self, SVDFile ):
		U, S, V = SVD.loadSVD(SVDFile)
		
		self.refineSVD = (U, S, V)
	
	def loadRefineVB( self, VBFile ):
		U, V = loadVB(VBFile)
		
		self.refineSVD = (U, None, V)
		
	def getMovieInfo( self, movie ):
		print "Movie ID:", movie
		print "Movie Title:", self.titles[movie-1]
		print "Movie Average:", self.movieAverages[movie-1]
		self.mmapFile[0].seek( self.mmapFile[1][movie-1] )
		str = self.mmapFile[0].read(struct.calcsize('I'))
		print "Number of ratings:", struct.unpack('I',str)[0]

	def findMovieID( self, title ):
		count = 0
		movies = []
		for t in self.titles:
			count = count + 1
			if re.compile(title).search(t)!=None:
				movies.append( (count, t) )
		return movies
	
	def getRating( self, user, movie ):
		print "User ID " + repr(user) + " is mapped to " + repr(self.userDict[user])
		userID = self.userDict[user]
		movieID = movie - 1
		return getRating( userID, movieID, self.mmapFile[0], self.mmapFile[1] )

	def findSimilarAnti( self, movie, simThreshold, antiThreshold ):
		return findSimilarAnti( movie, self.SVD[2], simThreshold, antiThreshold, self.norms )
		
	def printSimilarMovies( self, movie, simThreshold = 0.9, antiThreshold = -0.75 ):
		Prediction.getMovieInfo( self, movie )

		movieID = movie - 1
		similarMovies, antiMovies = Prediction.findSimilarAnti( self, movieID, simThreshold, antiThreshold )
		
		print "Similiar Movies"
	
		similarMovies = sorted(similarMovies, key=operator.itemgetter(1), reverse = True)

		for m in similarMovies:
			self.mmapFile[0].seek( self.mmapFile[1][m[0]] )
			str = self.mmapFile[0].read(struct.calcsize('I'))

			print m[1], "\t", self.movieAverages[m[0]], "\t", struct.unpack('I',str)[0], "\t", m[0]+1 ,"\t", self.titles[m[0]]
		
		print "\nAnti Movies"
		
		antiMovies = sorted(antiMovies, key=operator.itemgetter(1))	
		for m in antiMovies:
			self.mmapFile[0].seek( self.mmapFile[1][m[0]] )
			str = self.mmapFile[0].read(struct.calcsize('I'))

			print m[1], "\t", self.movieAverages[m[0]], "\t", struct.unpack('I',str)[0], "\t", m[0]+1 ,"\t", self.titles[m[0]]

	def predict( self, user, movie, verbose = True, mapped = False, refine = False, mode=None ):
		if verbose:
			print "User ID " + repr(user) + " is mapped to " + repr(self.userDict[user])
		if not mapped:
			userID = self.userDict[user]
			movieID = movie - 1
		else:
			userID = user
			movieID = movie

		if mode == None: mode = 'Normal'
		
		if mode not in ('Normal','Features'):
			print "Error: unknown mode:",mode
			return -1.
			
		if mode == 'Normal':
			if self.SVD[1] == None:
				pred = ((self.SVD[0])[userID,:] * (self.SVD[2])[movieID,:].T)[0,0]
			else:
				pred =  (((self.SVD[0])[userID] * (self.SVD[2])[movieID].T) + (self.SVD[0])[userID] * self.SVD[1] * (self.SVD[2])[movieID].T + self.intercept)[0,0]
			
			if refine:
				pred = max( pred, 1.0 )
				pred = min( pred, 5.0 )
				
				if self.refineSVD != None:
					if self.refineSVD[1] == None:
						pred = pred + ((self.refineSVD[0])[userID,:] * (self.refineSVD[2])[movieID,:].T)[0,0]
					else:
						pred = pred + ((self.refineSVD[0])[userID,:] * numpy.mat(numpy.diag(self.refineSVD[1])) * (self.refineSVD[2])[movieID,:].T)[0,0]
		else:
			features = []
			
			for r in xrange(rank):
				features.append( (self.SVD[0])[userID,r] * (self.SVD[2])[movieID,r] )
		
			for r in xrange(3):
				features.append( (self.SVD[0])[userID,r] )
				features.append( (self.SVD[2])[movieID,r] )			
				
			features = numpy.array(features)
			
			pred = numpy.dot( self.weights, features )
		
		pred = max( pred, 1.0 )
		pred = min( pred, 5.0 )
		
		return pred

	def predictFeatures( self, user, movie, dayOfWeek, week, year, holiday, verbose = True, mapped = False, refine = False, mode=None ):
		if verbose:
			print "User ID " + repr(user) + " is mapped to " + repr(self.userDict[user])
		if not mapped:
			userID = self.userDict[user]
			movieID = movie - 1
		else:
			userID = user
			movieID = movie

		if mode == None: mode = 'Normal'
		
		if mode not in ('Normal','Features'):
			print "Error: unknown mode:",mode
			return -1.
			
		if mode == 'Normal':
			if self.SVD[1] == None:
				pred = ((self.SVD[0])[userID,:] * (self.SVD[2])[movieID,:].T)[0,0]
			else:
				pred =  (((self.SVD[0])[userID] * (self.SVD[2])[movieID].T) + (self.SVD[0])[userID] * self.SVD[1] * (self.SVD[2])[movieID].T + self.intercept)[0,0]
			
			if refine:
				if self.refineSVD != None:
					if self.refineSVD[1] == None:
						pred = pred + ((self.refineSVD[0])[userID,:] * (self.refineSVD[2])[movieID,:].T)[0,0]
					else:
						pred = pred + ((self.refineSVD[0])[userID,:] * numpy.mat(numpy.diag(self.refineSVD[1])) * (self.refineSVD[2])[movieID,:].T)[0,0]
		else:
			features = []
			
			for r in xrange(rank):
				features.append( (self.SVD[0])[userID,r] * (self.SVD[2])[movieID,r] )
		
			for r in xrange(3):
				features.append( (self.SVD[0])[userID,r] )
				features.append( (self.SVD[2])[movieID,r] )			
				
			features = numpy.array(features)
			
			pred = numpy.dot( self.weights, features )
				
		if self.Z != None and self.W != None:
			if movieID+1 in self.imdbFeatures:
				featureList = [ f for f in self.imdbFeatures[movieID+1]]
			else:
				featureList = []
			featureList.append( dayOfWeek + self.dayOfWeekOffset )
			featureList.append( week + self.weekOffset )
			featureList.append( year-1 + self.yearOffset )
			if holiday == 1:
				featureList.append( self.holidayOffset )

			if self.Z != None:							
				preferences = (self.SVD[0])[userID,:]*self.Z
				pred = pred + numpy.sum(preferences[0,featureList])
	
			if self.W != None:
				pred = pred + numpy.sum(self.W[featureList]) 
		
		#pred = max( pred, 1.0 )
		#pred = min( pred, 5.0 )
		
		return pred
	
	def probe( self, refinement = False, limit = None, removeBadPredictions = False ):
		probeFile = open("probeRatingWithDate.dat", "rb")
		str = probeFile.read( struct.calcsize('>I') )
		e = struct.unpack('>I', str)
		totalEntries = e[0]
	
		if limit != None:
			totalEntries = limit
			
		print "probe Total number of entries:",totalEntries
	
		sumSquaredValues = 0.0
	
		if removeBadPredictions:
			sumBadSquaredValues = 0.0
			numBadEntries = 0
			
		numEntries = 0
		
		maxValue = -1.
		minValue = 10.
		
		for i in xrange(totalEntries):
			str = probeFile.read( struct.calcsize('>IHBBBBB') )
			e = struct.unpack('>IHBBBBB',str)
		
			movieID = e[1]
			userID = e[0]
			rating = e[2]
			dayOfWeek = e[3]
			week = e[4]
			year = e[5]
			holiday = e[6]
		
			# make prediction
			pred = Prediction.predictFeatures( self, userID, movieID, dayOfWeek, week, year, holiday, verbose = False, mapped = True, refine = refinement )
		
			maxValue = max( maxValue, pred )
			minValue = min( minValue, pred )
			
			actualPred = pred
			
			pred = max( pred, 1.0 )
			pred = min( pred, 5.0 )
			
			if removeBadPredictions:
				if abs( pred - float(rating) ) > 3.5:
					numBadEntries = numBadEntries + 1
					delta = float(rating) - pred
			
					sumBadSquaredValues = sumBadSquaredValues + delta * delta
					usualPred = False
				else:
					usualPred = True
			else:
				usualPred = True
				
			if usualPred:
				numEntries = numEntries + 1
				
				#print "rating:",rating,"pred:",pred
				delta = float(rating) - pred
			
				sumSquaredValues = sumSquaredValues + delta * delta 
		
		
		print repr(numEntries) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(numEntries)))
		
		print "Maximum Value:", maxValue
		print "Minimum Value:", minValue
		
		if removeBadPredictions:
			print repr(numBadEntries) + " bad pairs RMSE: " + repr(math.sqrt(sumBadSquaredValues/float(numBadEntries)))
			
		return math.sqrt(sumSquaredValues/float(numEntries))
	
	def qualifying( self, outputname, refinement = False, mode=None ):
		output = open( outputname, "w" )
		lastMovie = -1

		qualifying = open( "qualifyingMappedFeatures.dat", "rb" )

		str = qualifying.read( struct.calcsize('>I') )
		numQueries = (struct.unpack('>I',str))[0]

		for i in xrange(numQueries):
			str = qualifying.read( struct.calcsize('>IHBBBB') )
			e = struct.unpack('>IHBBBB',str)
		
			movieID = int(e[1])
			userID = int(e[0])
			dayOfWeek = int(e[2])
			week = int(e[3])
			year = int(e[4])
			holiday = int(e[5])

			#str = qualifying.read( struct.calcsize('>II') )
			#data = struct.unpack('>II',str)

			if lastMovie != movieID:
				if (movieID+1) % 100 == 0: print movieID+1
				output.write( repr(int(movieID+1)) + ":\n" )
				lastMovie = movieID

			pred = Prediction.predictFeatures( self, userID, movieID, dayOfWeek, week, year, holiday, verbose = False, mapped = True, refine = refinement, mode=mode )
			pred = max( pred, 1.0 )
			pred = min( pred, 5.0 )
			
			output.write( repr(pred) + "\n" )

		output.close()
		qualifying.close()

	def tweakQuiz( self, outputname, R = 0.9036, refinement = False, mode=None ):
		def tweakMean( R, predictedRatings, rMean, rVar ):
			QMean = 3.674405
			QVar = 1.274043
			
			A = QMean - rMean
			
			adjustedR = math.sqrt( (R*R) - (A*A) )
			
			adjustedRVar = rVar - A*A
			
			k = (QVar + adjustedRVar - (adjustedR*adjustedR))/(2 * adjustedRVar)
			
			print "Mean:", rMean
			print "Variance:", rVar
			print "Adjusted RVar:", adjustedRVar
			print "Adjusted^2 RVar:", (k*k) * adjustedRVar
	
			print "k:",k
			
			print "Predicted R is " + repr(math.sqrt( k*(adjustedR*adjustedR) + QVar*(1-k) - (k*(1-k)*adjustedRVar) ))
			
			return (R,k,A)

		def tweakVar( predictedRatings, k, A ):
			QMean = 3.674405
			numQueries = predictedRatings.shape[0]

			rMean = 0.
			rVar = 0.
			for i in xrange(numQueries):	
				predictedRatings[i] = k*((predictedRatings[i]+A) - QMean) + QMean
				rMean += predictedRatings[i]
				
				rVar += predictedRatings[i]*predictedRatings[i]
				
			rMean /= numQueries
			rVar /= numQueries
			
			rVar -= rMean * rMean
			
			return (rMean, rVar)
			
		#compute mean and variance
		qualifying = open( "qualifyingMappedFeatures.dat", "rb" )

		str = qualifying.read( struct.calcsize('>I') )
		numQueries = (struct.unpack('>I',str))[0]
		
		rMean = 0.
		rVar = 0.
		
		predictedRatings = numpy.zeros((numQueries,))
		
		for i in xrange(numQueries):
			#str = qualifying.read( struct.calcsize('>II') )
			#data = struct.unpack('>II',str)

			str = qualifying.read( struct.calcsize('>IHBBBB') )
			e = struct.unpack('>IHBBBB',str)
		
			movieID = int(e[1])
			userID = int(e[0])
			dayOfWeek = int(e[2])
			week = int(e[3])
			year = int(e[4])
			holiday = int(e[5])

			pred = Prediction.predictFeatures( self, userID, movieID, dayOfWeek, week, year, holiday, verbose = False, mapped = True, refine = refinement, mode=mode )

			predictedRatings[i] = pred
			
			rMean += pred
			
			rVar += pred*pred
			
		rMean /= numQueries
		rVar /= numQueries
		
		rVar -= rMean * rMean

		qualifying.close()
		
		for n in xrange(1):
			R, k, A = tweakMean( R, predictedRatings, rMean, rVar )
			rMean, rVar = tweakVar( predictedRatings, k, A )
		
		output = open( outputname, "w" )

		for i in xrange(numQueries):
			if i % 10000 == 0: print i
			
			pred = predictedRatings[i]
			
			pred = max(pred,1.0)
			pred = min(pred,5.0)
			
			output.write( repr(pred) + "\n" )

		output.close()
		
	def loadRegression( self, regressionFilename ):
		regressionFile = open( regressionFilename, "rb" )
		#str = regressionFile.read( struct.calcsize('I') )
		#rank = struct.unpack('I',str)[]
		rank = self.SVD[0].shape[1]
		str = regressionFile.read( struct.calcsize('d') )
		self.intercept = struct.unpack('d',str)[0]
		S = numpy.mat(numpy.zeros((rank,rank)))
		for r in xrange(rank):
			data = struct.unpack('d'*rank,str)
			for r in xrange(rank):
				S[u,r] = data[r]
		self.SVD[1] = S
		
	def loadLinear( self, regressionFilename ):
		regressionFile = open( regressionFilename, "rb" )
		rank = self.SVD[0].shape[1]
		str = regressionFile.read( struct.calcsize('>' +'d'*(rank*rank)) )
		lamda = numpy.zeros((rank*rank,))
		data = struct.unpack('>'+'d'*(rank*rank),str)
		for r in xrange(rank*rank):
				lamda[r] = data[r]
		S = numpy.mat(lamda.reshape(rank,rank))
		self.SVD = (self.SVD[0],S,self.SVD[2])
		
	def loadVB( self, VBFilename ):
		VBFile = open( VBFilename, "rb" )
		str = VBFile.read( struct.calcsize('>III') )
		r = struct.unpack('>III',str)
		numUsers = r[0]
		numMovies = r[1]
		rank = r[2]
		str = VBFile.read( struct.calcsize('>d') )
		#tau = struct.unpack('d',str)[0]

		str = VBFile.read( struct.calcsize('>'+'d' * rank) )

		PU = numpy.mat(numpy.zeros((numUsers,rank)))
		PV = numpy.mat(numpy.zeros((numMovies,rank)))
		for u in xrange(numUsers):
			str = VBFile.read( struct.calcsize('>'+'d'*rank) )
			data = struct.unpack('>'+'d'*rank,str)
			for r in xrange(rank):
				PU[u,r] = data[r]
		for m in xrange(numMovies):
			str = VBFile.read( struct.calcsize('>'+'d'*rank) )
			data = struct.unpack('>'+'d'*rank,str)
			for r in xrange(rank):
				PV[m,r] = data[r]
		VBFile.close()
	
		S = None
		self.SVD = (PU,S,PV)

	def loadFeatureVB( self, VBFilename ):	
		print "Loading hash..."
		#fID = open('movieID-to-features-25-cutoff.pickle', 'r')
		fID = open('movieID-to-features-map.pickle', 'r')
		self.imdbFeatures = pickle.load(fID)
		fID.close()

		m = -1
		maxNumFeatures = -1
		for k in self.imdbFeatures.keys():
			if len(self.imdbFeatures[k]) > maxNumFeatures: maxNumFeatures = len(self.imdbFeatures[k])
			for l in xrange(len(self.imdbFeatures[k])):
				if self.imdbFeatures[k][l] == 0: print "Error in feature list"
				self.imdbFeatures[k][l] -= 1 
				if self.imdbFeatures[k][l] > m:
					m = self.imdbFeatures[k][l]

		self.dayOfWeekOffset = m+1+0
		self.weekOffset = m+1+7
		self.yearOffset = m+1+61
		self.holidayOffset = m+1+68

		numFeatures = m+1+69
		VBFile = open( VBFilename, "rb" )
		str = VBFile.read( struct.calcsize('>III') )
		r = struct.unpack('>III',str)
		numUsers = r[0]
		numMovies = r[1]
		rank = r[2]
		str = VBFile.read( struct.calcsize('>d') )
		#tau = struct.unpack('d',str)[0]

		str = VBFile.read( struct.calcsize('>'+'d' * rank) )

		PU = numpy.mat(numpy.zeros((numUsers,rank)))
		PV = numpy.mat(numpy.zeros((numMovies,rank)))
		for u in xrange(numUsers):
			str = VBFile.read( struct.calcsize('>'+'d'*rank) )
			data = struct.unpack('>'+'d'*rank,str)
			for r in xrange(rank):
				PU[u,r] = data[r]
		for m in xrange(numMovies):
			str = VBFile.read( struct.calcsize('>'+'d'*rank) )
			data = struct.unpack('>'+'d'*rank,str)
			for r in xrange(rank):
				PV[m,r] = data[r]

		W = numpy.zeros((numFeatures,))

		W = numpy.zeros((numFeatures,))
		data = struct.unpack('>'+'d'*numFeatures, VBFile.read( struct.calcsize('>'+'d'*numFeatures) ) )
		for r in xrange(numFeatures):
			W[r] = data[r]
		
		if False:
			VBFile.read( struct.calcsize('>'+'d') )
	
			for r in xrange(rank):
				data = struct.unpack('>'+'d'*numFeatures, VBFile.read( struct.calcsize('>'+'d'*numFeatures) ) )
				for f in xrange(numFeatures):
					Z[r,f] = data[f]
			self.Z = Z
				
		VBFile.close()
	
		S = None
		self.SVD = (PU,S,PV)
		self.W = W
		
	def limitVB( self, rank ):
		U = numpy.mat(numpy.zeros((self.SVD[0].shape[0],rank)))
		V = numpy.mat(numpy.zeros((self.SVD[2].shape[0],rank)))
		
		if self.SVD[1] == None:
			U[:] = self.SVD[0][:,:rank]
			V[:] = self.SVD[2][:,:rank]
			self.SVD = (U,None,V)
		else:
			S = numpy.mat(numpy.zeros((rank,rank)))
			S[:] = self.SVD[1][:rank,:rank]
			self.SVD = (U,S,V)