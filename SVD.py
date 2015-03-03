import numpy
import scipy.linalg
import scipy.optimize
import math
import sys
import struct
import time
import scipy.weave
import operator
import scipy.stats

def secularEquation( p, residual, diagonal, borderSq ):
	total = residual - p - numpy.sum(borderSq/(diagonal-p))
	
	return total
	
def narrowInterval( a, b, residual, diagonal, borderSq ):
	phia = secularEquation( a, residual, diagonal, borderSq )
	phib = secularEquation( b, residual, diagonal, borderSq )
	if phia * phib < 0: return a,b
	
	#ta = a
	#tb = b
	while (1):
		#if a > ta and b < tb: break
		
		c = ( a + b ) / 2.0
		phic = secularEquation( c, residual, diagonal, borderSq )
		
		if phic == 0.0:
			a = c
			b = c
			break
		
		if phic < 0.0:
			b = c
			if phib * phic < 0: break
		else:
			a = c
			if phia * phic < 0: break
			
	return a, b
	
# given brokenarrow matrix, compute SVD
def brokenArrowSVD( M ):
	#bA = M.T * M	
	n = M.shape[0]
	
	diagonalM = numpy.array(numpy.diag(M[:n-1,:n-1]).flatten(1))
	diagonal = diagonalM**2
	border = diagonalM*numpy.array(M[:n-1,n-1].flatten(1))
	residual = M[:,n-1].T*M[:,n-1]

	absBorder = numpy.fabs(border)
	sumBorder = numpy.sum(absBorder)
	
	d = numpy.zeros((n+2,))
	d[1:n] = diagonal
	d[n] = (diagonal - absBorder).max()
	d[0] = (diagonal + absBorder).max()
			
	d[n] = min(d[n], residual - sumBorder)
	d[0] = max(d[0], residual + sumBorder)

	S = numpy.zeros(n,)
		
	borderSq = border**2
	border = border.flatten(1)
	
	V = numpy.mat(numpy.zeros((n,n)))
	
	for i in xrange( n ):
		a, b = narrowInterval( d[i+1], d[i], residual, diagonal, borderSq ) 
	
		root = scipy.optimize.brentq( secularEquation, a, b, args = ( residual, diagonal, borderSq ) )
	
		S[i] = math.sqrt(root)
		
		#V[i,:n-1] = border / ( root - diagonal )
		#V[i,n-1] = 1.0
		#V[i,:] = V[i,:]/scipy.linalg.norm(V[i,:])
		
		total = 0.0
		for j in xrange(n-1):
			V[i,j] = border[j]/(root - diagonal[j])
			total = total + V[i,j]*V[i,j]	
		V[i,n-1] = 1.0
		V[i,:] = V[i,:]/math.sqrt(total+1)

	Sinv = 1/S
	
	U = M * V.T * numpy.mat(numpy.diag(Sinv))
	return U, S, V
	
# loads SVD from filename and returns U, S, V
def loadSVD( filename ):
	svdFile = open( filename, "rb") #"svd-noprobe.dat", "rb")
	str = svdFile.read( struct.calcsize('>III') )
	dim = struct.unpack( '>III', str )

	numUsers = dim[0]
	numMovies = dim[1]
	rank = dim[2]

	print "Loading SVD\nnumUsers:"+repr(numUsers)+"\tnumMovies:"+repr(numMovies)+"\trank:"+repr(rank)

	U = numpy.mat(numpy.zeros((numUsers,rank)))
	V = numpy.mat(numpy.zeros((numMovies,rank)))
	S = numpy.zeros((rank,))
	
	for u in xrange(numUsers):
		str = svdFile.read( struct.calcsize('>'+'d'*rank) )
		e = struct.unpack( '>'+'d'*rank, str )
		for r in xrange(rank):
			U[u,r] = e[r]
		
	str = svdFile.read( struct.calcsize('>'+'d'*rank) )
	e = struct.unpack( '>'+'d'*rank, str )
	for r in xrange(rank):
		S[r] = e[r]
		
	for m in xrange(numMovies):
		str = svdFile.read( struct.calcsize('>'+'d'*rank) )
		e = struct.unpack( '>'+'d'*rank, str )
		for r in xrange(rank):
			V[m,r] = e[r]

	svdFile.close()
	print "Finished loading SVD"

	return U, S, V

def probeRMSE( U, S, V, mode = None, maxUserID=480189, maxMovieID=17770 ):
	if mode == None:
		mode = 'Normal'
		
	if mode not in ('Normal','Normalized'):
		print 'Unknown mode: ' + repr(mode)
		return
		
	if mode == 'Normalized':
		numMovies = 17770
		movieAverages = numpy.zeros((numMovies,))
		movieStdDev = numpy.zeros((numMovies,))
		movieAvgFile = open('movieAveragesStdDevNoProbe.dat','rb')
		for m in xrange(17770):
			str = movieAvgFile.read(struct.calcsize('dd'))
			data = struct.unpack('dd',str)
			movieAverages[m] = data[0]
			movieStdDev[m] = data[1]
		movieAvgFile.close()
	
	probeFile = open("probeRatingWithDate.dat", "rb")
	str = probeFile.read( struct.calcsize('>I') )
	e = struct.unpack('>I', str)
	totalEntries = e[0]

	print "probe Total number of entries:",totalEntries

	sumSquaredValues = 0.0

	count = 0
	
	for i in xrange(totalEntries):
		str = probeFile.read( struct.calcsize('>IHBBBBB') )
		e = struct.unpack('>IHBBBBB',str)
	
		movieID = e[1]
		userID = e[0]
		rating = e[2]
		#dayOfWeek = e[3]
		#week = e[4]
		#year = e[5]
		#holiday = e[6]
	
		if userID >= maxUserID: continue
		if movieID >= maxMovieID: continue
		
		# make prediction
		pred = (U[userID,:]*numpy.mat(numpy.diag(S))*V[movieID,:].T)[0,0]
		if mode == 'Normalized':
			pred = (pred * movieStdDev[movieID]) + movieAverages[movieID]
			
		pred = max( pred, 1.0 )
		pred = min( pred, 5.0 )
		
		delta = float(rating) - pred
	
		sumSquaredValues = sumSquaredValues + delta * delta 
		
		count = count + 1
	print repr(count) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(count)))
	
	return math.sqrt(sumSquaredValues/float(count))
	
def probeRMSE_VB( U, V, mode = None ):
	if mode == None:
		mode = 'Normal'
		
	if mode not in ('Normal','Normalized'):
		print 'Unknown mode: ' + repr(mode)
		return
		
	if mode == 'Normalized':
		numMovies = 17770
		movieAverages = numpy.zeros((numMovies,))
		movieStdDev = numpy.zeros((numMovies,))
		movieAvgFile = open('movieAveragesStdDevNoProbe.dat','rb')
		for m in xrange(17770):
			str = movieAvgFile.read(struct.calcsize('dd'))
			data = struct.unpack('dd',str)
			movieAverages[m] = data[0]
			movieStdDev[m] = data[1]
		movieAvgFile.close()
	
	probeFile = open("probeRatingWithDate.dat", "rb")
	str = probeFile.read( struct.calcsize('I') )
	e = struct.unpack('I', str)
	totalEntries = e[0]

	print "probe Total number of entries:",totalEntries

	sumSquaredValues = 0.0

	for i in xrange(totalEntries):
		str = probeFile.read( struct.calcsize('IHBBBBB') )
		e = struct.unpack('IHBBBBB',str)
	
		movieID = e[1]
		userID = e[0]
		rating = e[2]
		#dayOfWeek = e[3]
		#week = e[4]
		#year = e[5]
		#holiday = e[6]
	
		# make prediction
		pred = (U[userID,:]*V[movieID,:].T)[0,0]
		if mode == 'Normalized':
			pred = (pred * movieStdDev[movieID]) + movieAverages[movieID]
			
		pred = max( pred, 1.0 )
		pred = min( pred, 5.0 )
		
		delta = float(rating) - pred
	
		sumSquaredValues = sumSquaredValues + delta * delta 
	
	print repr(totalEntries) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(totalEntries)))
	
	return math.sqrt(sumSquaredValues/float(totalEntries))

def probeRMSE_VB_Features( U, V, mode = None ):
	if mode == None:
		mode = 'Normal'
		
	if mode not in ('Normal','Normalized'):
		print 'Unknown mode: ' + repr(mode)
		return
		
	if mode == 'Normalized':
		numMovies = 17770
		movieAverages = numpy.zeros((numMovies,))
		movieStdDev = numpy.zeros((numMovies,))
		movieAvgFile = open('movieAveragesStdDevNoProbe.dat','rb')
		for m in xrange(17770):
			str = movieAvgFile.read(struct.calcsize('dd'))
			data = struct.unpack('dd',str)
			movieAverages[m] = data[0]
			movieStdDev[m] = data[1]
		movieAvgFile.close()
	
	probeFile = open("probeRatingWithDate.dat", "rb")
	str = probeFile.read( struct.calcsize('I') )
	e = struct.unpack('I', str)
	totalEntries = e[0]

	print "probe Total number of entries:",totalEntries

	sumSquaredValues = 0.0
	
	weights = numpy.array([  1.10552043e+00,   1.06280562e+00,   1.16521195e+00,   1.15557848e+00, 1.14855813e+00,   1.24595619e+00,   1.24694138e+00,   1.20906510e+00, 1.28430852e+00,   1.27358478e+00,   1.35120293e+00,   1.47781860e+00, 1.34472806e+00,   1.46486818e+00,   1.28884820e+00,   1.40381748e+00, 1.42082280e+00,   1.43480485e+00,   1.69116145e+00,   1.56163830e+00, 1.46557682e+00,  1.52025822e+00,   1.61233520e+00,   1.45983051e+00, 1.65578358e+00,   1.32226923e+00,   1.69972770e+00,   1.84679831e+00, 1.69446285e+00,   1.46407733e+00,   1.89950253e+00,   1.86899257e+00, 1.84986662e+00,   1.94316118e+00,   1.60390989e+00,   1.75202649e+00, 1.76281290e+00,   2.00891981e+00,   1.95296822e+00,   2.19943889e+00, 2.03976220e+00,   2.08147846e+00,   2.00504910e+00,   2.09129370e+00, 1.80114910e+00,   1.96831021e+00,   2.16509513e+00,   2.14036753e+00, 1.27838317e+00,   2.20098761e+00,  -3.09331588e-05,  -3.02321588e+02, 2.70448530e-05,  -1.13985603e+01,  7.16042656e-05,  -9.99935172e+00])
			
	rank = U.shape[1]
	
	for i in xrange(totalEntries):
		str = probeFile.read( struct.calcsize('IHBBBBB') )
		e = struct.unpack('IHBBBBB',str)
	
		movieID = e[1]
		userID = e[0]
		rating = e[2]
		#dayOfWeek = e[3]
		#week = e[4]
		#year = e[5]
		#holiday = e[6]
	
		# make prediction   
		features = []
				
		for r in xrange(rank):
			features.append( U[userID,r] * V[movieID,r] )
			
		for r in xrange(3):
			features.append( U[userID,r] )
			features.append( V[movieID,r] )			
					
		features = numpy.array(features)
				
		pred = numpy.dot( weights, features )
	
		pred = max( pred, 1.0 )
		pred = min( pred, 5.0 )
		
		delta = float(rating) - pred
	
		sumSquaredValues = sumSquaredValues + delta * delta 
	
	print repr(totalEntries) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(totalEntries)))
	
	return math.sqrt(sumSquaredValues/float(totalEntries))
	
def probeRMSERefine( U, S, V, approxU, approxS, approxV, mode = None ):
	if mode == None:
		mode = 'Normal'
		
	if mode not in ('Normal','Normalized'):
		print 'Unknown mode: ' + repr(mode)
		return
		
	if mode == 'Normalized':
		numMovies = 17770
		movieAverages = numpy.zeros((numMovies,))
		movieStdDev = numpy.zeros((numMovies,))
		movieAvgFile = open('movieAveragesStdDevNoProbe.dat','rb')
		for m in xrange(17770):
			str = movieAvgFile.read(struct.calcsize('dd'))
			data = struct.unpack('dd',str)
			movieAverages[m] = data[0]
			movieStdDev[m] = data[1]
		movieAvgFile.close()
	
	probeFile = open("probeRatingWithDate.dat", "rb")
	str = probeFile.read( struct.calcsize('I') )
	e = struct.unpack('I', str)
	totalEntries = e[0]

	print "probe Total number of entries:",totalEntries

	sumSquaredValues = 0.0

	for i in xrange(totalEntries):
		str = probeFile.read( struct.calcsize('IHBBBBB') )
		e = struct.unpack('IHBBBBB',str)
	
		movieID = e[1]
		userID = e[0]
		rating = e[2]
		#dayOfWeek = e[3]
		#week = e[4]
		#year = e[5]
		#holiday = e[6]
	
		# make prediction
		if approxS == None:
			pred = (approxU[userID,:]*approxV[movieID,:].T)[0,0] + (U[userID,:]*numpy.mat(numpy.diag(S))*V[movieID,:].T)[0,0]
		else:	
			pred = (approxU[userID,:]*numpy.mat(numpy.diag(approxS))*approxV[movieID,:].T)[0,0] + (U[userID,:]*numpy.mat(numpy.diag(S))*V[movieID,:].T)[0,0]
		
		if mode == 'Normalized':
			pred = (pred * movieStdDev[movieID]) + movieAverages[movieID]
			
		pred = max( pred, 1.0 )
		pred = min( pred, 5.0 )
		
		delta = float(rating) - pred
	
		sumSquaredValues = sumSquaredValues + delta * delta 
	
	print repr(totalEntries) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(totalEntries)))
	
	return math.sqrt(sumSquaredValues/float(totalEntries))
	
# returns U, S, V
def incrementalSVD( M ):
	mrank = M.shape[1]
	TU = numpy.mat(numpy.zeros(M.shape))
	TV = numpy.mat(numpy.zeros((0,mrank)))
	TS = numpy.zeros((mrank,))

	numUsers = M.shape[0]
	
	#print M
	for column in xrange(mrank):		
		TL = TU.T * M[:,column] 
		TT = TU * TL
		
		if numUsers < 6000:
			TcT = M[:,column] - TT
			Tk = scipy.linalg.norm(TcT)
			TJ = (TcT)/Tk
		else:
			TT = TT.A
			c = M[:,column].A
			TJ = numpy.zeros((numUsers,1))
			code = """
				double total, diff, norm;
				total = 0.0;
				for ( int i = 0; i < numUsers; ++i ) {
					diff = c(i,0) - TT(i,0);
					total += diff * diff;
					TJ(i,0) = diff;
				}
				norm = sqrt(total);
				for (int i = 0; i < numUsers; ++i )
					TJ(i,0) /= norm;
			
				return_val = norm;
				"""
			Tk = scipy.weave.inline(code, ['numUsers','TJ','c','TT'], type_converters = scipy.weave.converters.blitz, compiler = 'gcc' )		
				
		TQ = numpy.mat(numpy.zeros((mrank+1,mrank+1))) 
		for l in xrange(mrank):
			TQ[l,l] = TS[l]
		TQ[:mrank,mrank] = TL
		TQ[mrank,mrank]=Tk
		
		TUp,TSp,TVp = scipy.linalg.svd(TQ,overwrite_a=1)
	
		TU = numpy.concatenate((TU,TJ),1)*TUp
		TU = TU[:,:mrank]
		TS = TSp[:mrank]
		
		TVt = numpy.mat(numpy.zeros((column+1,mrank+1)))
		TVt[:column,:mrank]=TV
		TVt[column,mrank]=1
		
		TV = TVt * TVp.T
		TV = TV[:,:mrank]

		if column > 0 and math.fabs(TU[:,0].T * TU[:,-1:]) > 1e-3:
			q,r = scipy.linalg.qr(TU,overwrite_a=1,econ=True)
			TU = numpy.mat(q) * numpy.mat(numpy.diag(numpy.diag(r)/numpy.fabs(numpy.diag(r))))
	#print TU * numpy.mat(numpy.diag(TS)) * TV.T
	return TU,TS,TV
	
def testRMSE( U, S, V, userMap, userAddress, movieMap, movieAddress, maxRatings ):
	numMovies = 17770
	movieAverages = numpy.zeros((numMovies,))
	movieStdDev = numpy.zeros((numMovies,))
	movieAvgFile = open('movieAveragesStdDevNoProbe.dat','rb')
	for m in xrange(17770):
		str = movieAvgFile.read(struct.calcsize('dd'))
		data = struct.unpack('dd',str)
		movieAverages[m] = data[0]
		movieStdDev[m] = data[1]
	movieAvgFile.close()
	
	probeFile = open("probeRatingWithDate.dat", "rb")
	str = probeFile.read( struct.calcsize('I') )
	e = struct.unpack('I', str)
	totalEntries = e[0]

	print "probe Total number of entries:",totalEntries

	sumSquaredValues = 0.0
	sumSquaredValues_median = 0.0
	sumSquaredValues_movieMean = 0.0
	sumSquaredValues_mean = 0.0
	sumSquaredValues_mode = 0.0
	sumSquaredValues_weighted = 0.0
	sumSquaredValues_weightedInv = 0.0
	sumSquaredValues_weightedSmart = 0.0
	sumSquaredValues_weightedSmart2 = 0.0
	sumSquaredValues_weightedSmart3 = 0.0

	minMatchedRatings = 1

	log = open( "RMSE.log", "a" )
	log.write( "Num Ratings:"+repr(maxRatings)+"\n" )
	
	num = 0
	for i in xrange(totalEntries):
		if i % 100000 == 0: 
			print "Probe Entry:",i
			log.write("Probe Entry Number:" + repr(i) + "\n")
		
		str = probeFile.read( struct.calcsize('IHBBBBB') )
		e = struct.unpack('IHBBBBB',str)
	
		movieID = e[1]
		userID = e[0]
		rating = e[2]
		#dayOfWeek = e[3]
		#week = e[4]
		#year = e[5]
		#holiday = e[6]
	
		# number of ratings
		userMap.seek( userAddress[ userID ] )
		str = userMap.read(struct.calcsize('H'))
		numRatings = struct.unpack('H',str)[0]

		if numRatings > maxRatings:
			continue
			
		ratings = []
		moviesRated = []

		# get all user ratings
		for l in xrange(numRatings):
			R = struct.unpack('HB',userMap.read(struct.calcsize('HB')))
			
			# get num ratings of movie
			movieMap.seek( movieAddress[ R[0] ] )
			str = movieMap.read(struct.calcsize('I'))
			numMovieRatings = struct.unpack('I',str)[0]
			
			ratings.append((R[0],numMovieRatings,R[1]))
			moviesRated.append(R[0])

		#ratings = sorted( ratings, key=operator.itemgetter(1))#, reverse = True )

		# get all other users with same rating from movies
		matched = []
		for m in xrange(len(ratings)):
			movieMap.seek( movieAddress[ ratings[m][0] ] )
			str = movieMap.read(struct.calcsize('I'))
			for l in xrange(ratings[m][1]):
				R = struct.unpack('IB',movieMap.read(struct.calcsize('IB')))
				if R[1] == ratings[m][2] and R[0] != userID:
					matched.append(R[0])

		matched = set(matched)
		
		# use user ratings to match to user
		newMatched = []
		for m in matched:
			allSame = True
			userMap.seek( userAddress[ m ] )
			str = userMap.read(struct.calcsize('H'))
			length = struct.unpack('H',str)[0]
			for l in xrange(length):
				R = struct.unpack('HB',userMap.read(struct.calcsize('HB')))
				if R[0] in moviesRated:
					# check that user has the same rating
					for k in xrange(len(ratings)):
						if ratings[k][0] == R[0]:
							if ratings[k][2] != R[1]:
								allSame = False
							break

					if not allSame: break
			if allSame: newMatched.append(m)

		matched = newMatched

		if len(matched) < minMatchedRatings: continue
		
		# now get ratings for movie of users matched
		matchedRatings = []
		movieMap.seek( movieAddress[ movieID ] )
		str = movieMap.read(struct.calcsize('I'))
		length = struct.unpack('I',str)[0]
	
		matchedRatingsCount = []
		for l in xrange(length):
			R = struct.unpack('IB',movieMap.read(struct.calcsize('IB')))
			if R[0] in matched:
				matchedRatings.append( R[1] )
				userMap.seek( userAddress[ R[0] ] )
				str = userMap.read(struct.calcsize('H'))
				matchedRatingsCount.append( struct.unpack('H',str)[0]	)
				
		if len(matchedRatings) < minMatchedRatings: continue

		print "numRatings:",numRatings,"\tmovieid:",movieID,"\tnumMovieRatings:",length,"\tnumMatchedRatings:",len(matchedRatings),"\trating:",rating,"\tavg:", movieAverages[movieID],"\tstd_dev:", movieStdDev[movieID],"\t",

		log.write( "numRatings:" + repr(numRatings) + "\tmovieid:" + repr(movieID) + "\tnumMovieRatings:" +repr(length) + "\tnumMatchedRatings:" + repr(len(matchedRatings)) + "\trating:" + repr(rating) + "\tavg:" +  repr(movieAverages[movieID]) + "\tstd_dev:" + repr(movieStdDev[movieID]) + "\t")
		matchedRatings = numpy.array(matchedRatings)
		
		num = num + 1
		
		# make SVD prediction
		pred = (U[userID,:]*numpy.mat(numpy.diag(S))*V[movieID,:].T)[0,0]
		pred = (pred * movieStdDev[movieID]) + movieAverages[movieID]
			
		pred = max( pred, 1.0 )
		pred = min( pred, 5.0 )
		
		delta = float(rating) - pred
	
		log.write("SVD:" + repr(pred) + "\t")
		print "SVD:",pred,"\t",
		sumSquaredValues = sumSquaredValues + delta * delta 

		# movie average
		pred = movieAverages[movieID]
		delta = float(rating) - pred
	
		sumSquaredValues_movieMean = sumSquaredValues_movieMean + delta * delta 
		
		# mean prediction
		pred = matchedRatings.mean()
		delta = float(rating) - pred
	
		log.write("mean:" + repr(pred) + "\t")
		print "mean:",pred,"\t",
		sumSquaredValues_mean = sumSquaredValues_mean + delta * delta 

		# median prediction
		pred = numpy.median( matchedRatings )
		delta = float(rating) - pred
	
		log.write("median:" + repr(pred) + "\t")
		print "median:",pred,"\t",
		sumSquaredValues_median = sumSquaredValues_median + delta * delta 

		# mode prediction
		pred = scipy.stats.mode( matchedRatings )[0][0]
		delta = float(rating) - pred
	
		log.write("mode:" + repr(pred) + "\t")
		print "mode:",pred,"\t",
		sumSquaredValues_mode = sumSquaredValues_mode + delta * delta 
		
		#weighted prediction
		pred = 0.
		total = 0
		for l in xrange(len(matchedRatings)):
			pred = pred + matchedRatings[l] * matchedRatingsCount[l]
			total = total + matchedRatingsCount[l]
		
		pred = pred / float(total)
		delta = float(rating) - pred

		log.write( "weighted:" + repr(pred) + "\t")
		print "weighted:",pred
	
		sumSquaredValues_weighted = sumSquaredValues_weighted + delta * delta 

		if len(matchedRatings)<50:
			pred = movieAverages[movieID]
			delta = float(rating) - pred

		log.write( "weighted smart:" + repr(pred) + "\t")

		sumSquaredValues_weightedSmart = sumSquaredValues_weightedSmart + delta*delta

		if len(matchedRatings)<50:
			pred = matchedRatings.mean()
			delta = float(rating) - pred

		log.write( "weighted smart 2:" + repr(pred) + "\n")

		sumSquaredValues_weightedSmart2 = sumSquaredValues_weightedSmart2 + delta*delta

		if len(matchedRatings) == 1:
			pred = movieAverages[movieID]
		elif len(matchedRatings)<50:
			pred = matchedRatings.mean()
			
		delta = float(rating) - pred

		log.write( "weighted smart 3:" + repr(pred) + "\n")

		sumSquaredValues_weightedSmart3 = sumSquaredValues_weightedSmart3 + delta*delta

		log.flush()

		#weighted inverse prediction
		pred = 0.
		totalMatchedRatingsCount = total
		total = 0
		for l in xrange(len(matchedRatings)):
			pred = pred + matchedRatings[l] * (totalMatchedRatingsCount+1-matchedRatingsCount[l])
			total = total + totalMatchedRatingsCount+1-matchedRatingsCount[l]
		
		pred = pred / float(total)
		delta = float(rating) - pred
	
		sumSquaredValues_weightedInv = sumSquaredValues_weightedInv + delta * delta 
		
		if num % 100 == 0:
			print "\tNum valid:", num, "\tSVD:", math.sqrt(sumSquaredValues/float(num)), "\tMovie Mean:", math.sqrt(sumSquaredValues_movieMean/float(num)), "\tMean:", math.sqrt(sumSquaredValues_mean/float(num)), "\tMedian:", math.sqrt(sumSquaredValues_median/float(num)), "\tMode:", math.sqrt(sumSquaredValues_mode/float(num))
			print "\tWeighted:", math.sqrt(sumSquaredValues_weighted/float(num)), "\tWeightInv:", math.sqrt(sumSquaredValues_weightedInv/float(num)), "\tWeighted smart:", math.sqrt(sumSquaredValues_weightedSmart/float(num)), "\tWeight smart 2:", math.sqrt(sumSquaredValues_weightedSmart2/float(num))

	print "\nSVD"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(num)))
	print "Movie Mean"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_movieMean/float(num)))
	print "mean"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_mean/float(num)))
	print "median"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_median/float(num)))
	print "mode"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_mode/float(num)))
	print "weighted"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weighted/float(num)))
	print "weighted smart"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weightedSmart/float(num)))
	print "weightedInv"
	print repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weightedInv/float(num)))

	log.write( "\nSVD\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(num))) )
	log.write( "Movie Mean\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_movieMean/float(num))) ) 
	log.write( "mean\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_mean/float(num))))
	log.write( "median\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_median/float(num))))
	log.write( "mode\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_mode/float(num))))
	log.write( "weighted\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weighted/float(num))))
	log.write( "weighted smart\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weightedSmart/float(num))))
	log.write( "weighted smart 2\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weightedSmart2/float(num))))
	log.write( "weighted smart 3\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weightedSmart3/float(num))))
	log.write( "weightedInv\n" )
	log.write( repr(num) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues_weightedInv/float(num)))+"\n" )
	log.close()

	return math.sqrt(sumSquaredValues/float(num))
