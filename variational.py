import numpy
import scipy.linalg
import scipy.stats
import scipy.weave
import pickle
import random
import math
import sys
import struct
import time
import datetime
import SVD
import prediction

# ---------------------------
# INITIALIZATION STUFF STARTS
# ---------------------------

def matmult(a,b, alpha=1.0, beta=0.0, c=None, trans_a=0, trans_b=0):
	"""Return alpha*(a*b) + beta*c.
	a,b,c : matrices
	alpha, beta: scalars
	trans_a : 0 (a not transposed), 1 (a transposed), 2 (a conjugate transposed)
	trans_b : 0 (b not transposed), 1 (b transposed), 2 (b conjugate transposed)
	"""
	if c != None:
		gemm,=scipy.linalg.get_blas_funcs(('gemm',),(a,b,c))
	else:
		gemm,=scipy.linalg.get_blas_funcs(('gemm',),(a,b))

	return gemm(alpha, a, b, beta, c, trans_a, trans_b)

numUsers = 480189
#numUsers = 20000
numMovies = 17770
trainingMode = 'Residual' # 'Full', 'NoProbe', 'Residual'

desired = 50

# initialize by loading SVD
loadSVD = True

if trainingMode == 'NoProbe':
	ratingFilename = 'userMovieNoProbe.dat'
elif trainingMode == 'Full':
	ratingFilename = 'userMovie.dat'
elif trainingMode == 'Residual':
	ratingFilename = 'residualUserMovie.dat'
	
ratingFile = open(ratingFilename,'rb')
str = ratingFile.read(struct.calcsize('>I'))
u = struct.unpack('>I',str)[0]

if u != numUsers:
	print "Error: read in", u, "as numUsers =", numUsers
	#sys.exit(1)
	
str = ratingFile.read(struct.calcsize('>I'))
totalEntries = struct.unpack('>I',str)[0]

print "Total number of entries:", totalEntries
ratingFile.close()

if trainingMode == 'Residual':
	ratingFile, startAddress = prediction.mmapData( ratingFilename, mode = "ResidualUserMovie", addresses = "StartOnly" )
else:
	ratingFile = open(ratingFilename,'rb')
	ratingFile.read(struct.calcsize('>I'))
	ratingFile.read(struct.calcsize('>I'))

if loadSVD:
	if trainingMode == 'Residual':
		PU, PS, PV = SVD.loadSVD("svd-residual-probe-em-1-work-50-Asc-50rank.dat")
	else:
		PU, PS, PV = SVD.loadSVD("svd-probe.dat")
		#PU, PS, PV = SVD.loadSVD("svd-probe.0.6636.dat")

	rank = int(PU.shape[1])

	PU = PU * numpy.mat(numpy.diag(PS))
	del PS

	if desired < rank:
		rank = desired
		TU = numpy.mat(numpy.zeros((numUsers,rank)))
		TV = numpy.mat(numpy.zeros((numMovies,rank)))
		TU[:] = PU[:,:rank]
		TV[:] = PV[:,:rank]
		del PU
		del PV
		PU = TU
		PV = TV
	
	# transform PU and PV
	norms = numpy.diag(PV.T * PV)
	for c in xrange(norms.shape[0]):
		PV[:,c] = PV[:,c] / math.sqrt( norms[c] * rank )
		PU[:,c] = PU[:,c] * math.sqrt( norms[c] * rank )		
	del norms
	
	#rho = numpy.ones((rank,)) * 1/float(rank)
	sigma = numpy.zeros((rank,))
	
	print "Initialize sigma...",
	for r in xrange(rank):
		sigma[r] = (PU[:,r].T * PU[:,r])[0,0]
		sigma[r] = sigma[r] / float(numUsers-1)
	print "done"
	
	# do not need PU
	del PU
	
	tau = 1.
	#tau = 0.7

	# covariance of PV
	covPV = numpy.zeros((numMovies,rank,rank))
		
else:
	# load from a previous VB data file
	if trainingMode == 'Residual':
		VBFilename = "VB-residual.dat"
	else:
		#VBFilename = "VB-rank-100.0.9066.dat"
		#VBFilename = "VB.dat"
		VBFilename = "yjvb.dat"
	
	VBFile = open( VBFilename, "rb" )
	str = VBFile.read( struct.calcsize('>III') )
	r = struct.unpack('>III',str)
	numUsers = int(r[0])
	numMovies = int(r[1])
	rank = int(r[2])
	str = VBFile.read( struct.calcsize('>d') )
	tau = struct.unpack('>d',str)[0]
	#tau = 1.
	
	sigma = numpy.zeros((rank,))
	str = VBFile.read( struct.calcsize('>'+'d' * rank) )
	data = struct.unpack('d'*rank,str)
	for r in xrange(rank):
		sigma[r] = data[r]

	PU = numpy.mat(numpy.zeros((numUsers,rank)))
	PV = numpy.mat(numpy.zeros((numMovies,rank)))
	for u in xrange(numUsers):
		str = VBFile.read( struct.calcsize('>'+'d'*rank) )
		if desired != r:
			data = struct.unpack('>'+'d' * rank,str)
			for r in xrange(rank):
				PU[u,r] = data[r]
	
	for m in xrange(numMovies):
		str = VBFile.read( struct.calcsize('>'+'d'*rank) )
		data = struct.unpack('>'+'d' * rank,str)
		for r in xrange(rank):
			PV[m,r] = data[r]
	
	if desired != rank:
		rank = desired
		TU = numpy.mat(numpy.zeros((numUsers,rank)))
		TV = numpy.mat(numpy.zeros((numMovies,rank)))
		TU[:] = PU[:,:rank]
		TV[:] = PV[:,:rank]
		del PU
		del PV
		PU = TU
		PV = TV
		sigma = numpy.zeros((rank,))
		tau = 1.
		norms = numpy.diag(PV.T * PV)
		for c in xrange(norms.shape[0]):
			PV[:,c] /= math.sqrt( norms[c] * rank )
			PU[:,c] *= math.sqrt( norms[c] * rank )	
		del norms		
		
		print "Initialize sigma...",
		for r in xrange(rank):
			sigma[r] = (PU[:,r].T * PU[:,r])[0,0]
			sigma[r] = sigma[r] / float(numUsers-1)
		print "done"
		del PU

	print tau
	print sigma
	covPV = numpy.zeros((numMovies,rank,rank))
	
	#for m in xrange(numMovies):
	#	for r in xrange(rank):
	#		str = VBFile.read( struct.calcsize('d'*rank) )
	#		data = struct.unpack('d' * rank,str)
	#		for k in xrange(rank):
	#			covPV[m,r,k] = data[k]

	#rho = numpy.ones((rank,)) * 1/float(rank)

	#str = VBFile.read( struct.calcsize('d' * rank) )
	#data = struct.unpack('d'*rank,str)
	#for r in xrange(rank):
	#	rho[r] = data[r]
	VBFile.close()

iter = 0

#We eliminate the use of S and replace storage by using covPV
#covPV will initially store the previous iteration's Psi, i.e. covariance matrix
#OuterV will use covPV for precomputation of V.T*V + covPV
#covPV is then initialized to 1/rho

#outerV = numpy.zeros((rank,rank))
outerV = numpy.zeros((numMovies,rank,rank))
#S = numpy.zeros((numMovies,rank,rank))
tempMovieID = numpy.zeros((numMovies,),numpy.int16)
tempRatings = numpy.zeros((numMovies,))
	
mater = numpy.mat
unpacker = struct.unpack
diager = numpy.diag
HSize = struct.calcsize('H')
if trainingMode == 'Residual':
	IdSize = struct.calcsize('>Id')
else:
	HBSize = struct.calcsize('>HB')
zeroser = numpy.zeros
inliner = scipy.weave.inline
inverter = scipy.linalg.inv
oneser = numpy.ones
blitzer = scipy.weave.converters.blitz
incS = zeroser((rank,rank))
outerU = zeroser((rank,rank))
covU = zeroser((rank,rank))
U = mater(zeroser((numUsers,rank)))
V = mater(zeroser((numMovies,rank)))

# -------------------------
# INITIALIZATION STUFF ENDS
# -------------------------

tau = 0.456

while True:

	iter = iter + 1
	print "Iteration: " + repr(iter) 
	
	U[:] = 0.
	V[:] = 0.
	
	if trainingMode == 'Residual':
		ratingFile.seek(startAddress)
		reader = ratingFile.read
	else:
		ratingFile.close()
		ratingFile = open(ratingFilename,'rb')
		ratingFile.read(struct.calcsize('>I'))
		ratingFile.read(struct.calcsize('>I'))
		reader = ratingFile.read
		
	# precompute outer product V.T * V + Psi
	for m in xrange(numMovies):
		# create a diagonal matrix with diagonal entries consisting of 1./rho
		#S[m] = diager( 1./rho )

		outerV[m] = ( (covPV[m] + PV[m,:].T * PV[m,:]) )
		#outerV[m] = matmult(PV[m,:].T,PV[m,:],beta=1.0,c=covPV[m])
		#outerV[m] = (PV[m,:].T * PV[m,:])

	for m in xrange(numMovies):
		# create a diagonal matrix with diagonal entries consisting of 1./rho
		#covPV[m] = diager( 1./rho )
		covPV[m] = diager( numpy.ones((rank,)) * rank )
		
	newSigma = zeroser((rank,))
	newTau = 0.
	
	entries = 0
	
	#PV = PV.A
	#tau = 3.
	
	for u in xrange(numUsers):
		if u % 10000 == 0: 
			if u > 0: 
				print time.asctime(),"\tUser ID ["+repr(u)+"/"+repr(numUsers)+"]","\t",newTau/float(entries-1)
			else:
				print time.asctime(),"\tUser ID ["+repr(u)+"/"+repr(numUsers)+"]"

		# read in number of ratings
		length = unpacker('>H',reader(HSize))[0]

		entries = entries + length

		# -------------------
		# VB STARTS FROM HERE
		# -------------------
		
		covU[:] = diager(1./sigma)
		#covU[:] = 0.
		
		if trainingMode == 'Residual':
			R = unpacker('>'+'Id'*length,reader(IdSize*length))
			for l in xrange(length):
				tempMovieID[l] = int(R[l<<1])
				tempRatings[l] = float(R[(l<<1)+1])
			del R
		else:
			for l in xrange(length):
				R = unpacker('>HB',reader(HBSize))
				movieID = int(R[0])
				rating = float(R[1])
				tempMovieID[l] = movieID
				tempRatings[l] = rating
				
			#covU = covU + outerV[movieID]
			#U[u] = U[u] + ( rating / tau ) * PV[movieID]

		# this is what the inline code does	
		#covU += numpy.sum(outerV[tempMovieID[:length]]/tau,axis=0)
		#U[u] += numpy.sum(numpy.multiply( ( tempRatings[:length] / tau ), PV[tempMovieID[:length]].T ).T,axis=0)
		
		TU = U[u].A
		TV = PV.A
		code = """
			short *pTempMovieID = tempMovieID;
			double *pTempRatings = tempRatings;
			double *pcovU = covU;
			double *pOuterV = outerV;
			double *pTU = TU;
			double *pTV = TV;
			
			int size = (int)rank;
			for ( int l = 0; l < length; ++l ) {
				int movieID = (int)*pTempMovieID;
				double rating = *pTempRatings;
				
				double mult = 0.0;

				pTV = TV;
				pTV += (movieID * size);
				
				pTU = TU;
				pcovU = covU;

				pOuterV = outerV;
				pOuterV += (movieID * size * size);
				
				for ( int i = 0; i < size; ++i ) {
					for ( int j = 0; j < size; ++j ) {
						*pcovU += (*pOuterV)/(double)tau;
						pcovU++; pOuterV++;
					}

					*pTU += rating * (*pTV);

					pTU++; pTV++;
				}
				
				pTempMovieID++; pTempRatings++;
			}
		"""
		inliner( code, ['length','tempMovieID','tempRatings','covU','TU','TV','rank','outerV','tau'], compiler = 'gcc' )
		
		U[u] = U[u] / tau
		
		# invert covariance matrix (Phi)
		covU = inverter(covU)
		
		#This is what the inline code does
		#U[u] = (covU * U[u].T).T
		###outerU = (U[u,:].T * U[u,:] + covU)
		newSigma += diager(covU)
		#incS = (outerU / tau).A
		#S[tempMovieID[:length]] = S[tempMovieID[:length]] + incS.A
		#V[tempMovieID[:length]] = V[tempMovieID[:length]] + (numpy.multiply( (tempRatings[:length]/tau), U[u].T ).T)		
		
		TU = U[u].A
		TTU = U[u].copy().A
		#outerU = covU
		#TV = V[tempMovieID[:length]].A
		TV = V.A
		TTV = PV.A
		
		code = """
			short *pTempMovieID = tempMovieID;
			double *pTempRatings = tempRatings;
			double *pcovU = covU;
			double *pcovPV = covPV;
			double *pOuterV = outerV;
			double *pOuterU = outerU;
			double *pTU = TU;
			double *pTTU = TTU;
			double *pTV = TV;
			double *pTTV = TTV;
			double *pIncS = incS;
			
			int size = (int)rank;

			for ( int i = 0; i < size; ++i ) {
				*pTU = 0.0;
				
				pTTU = TTU;
				for ( int j = 0; j < size; ++j ) {
					*pTU += (*pcovU) * (*pTTU);
					
					pcovU++; pTTU++;
				}
				
				pTU++;
			}
			
			pcovU = covU;
			pTU = TU;
			double *pTU2 = TU;
			
			for ( int i = 0; i < size; ++i ) {
				
				pTU2 = TU;
				for ( int j = 0; j < size; ++j ) {
					*pOuterU = (*pcovU) + ((*pTU) * (*pTU2) );
					
					pcovU++; pOuterU++; pTU2++;
				}
				
				pTU++;
			}				

			for ( int l = 0; l < length; ++l ) {
				int movieID = (int)*pTempMovieID;
				double rating = *pTempRatings;
				
				double mult = 0.0;
				
				pIncS = incS;
				pOuterU = outerU;
				
				pcovPV = covPV + movieID * size * size;
				pOuterV = outerV + movieID * size * size;
				
				pTV = TV + movieID * size;
				pTTV = TTV + movieID * size;
				pTU = TU;
				
				for ( int i = 0; i < size; ++i ) {
					for ( int j = 0; j < size; ++j ) {
						if ( l == 0 ) 
							*pIncS = (*pOuterU) / tau;

						*pcovPV += *pIncS;
						newTau += (*pOuterU) * (*pOuterV);
						
						pIncS++; pOuterU++; pcovPV++; pOuterV++;
					}

					*pTV += rating * (*pTU);
					mult += (*pTU) * (*pTTV);
					
					pTV++; pTTV++; pTU++;
				}

				/*
				double old = mult;
				
				if (mult<1.0) mult=1.0;
				if (mult>5.0) mult=5.0;
				
				double adj = mult-old;
				*/
				
				newTau += rating * ( rating - 2 * mult ); // + adj*adj + 2*adj*old;

				pTempMovieID++; pTempRatings++;
			}

			return_val = newTau;
		"""
		t = inliner( code, ['TTU','TU','rank','outerU','covU','length','tempMovieID','tempRatings','TV','TTV','tau','covPV','incS','newTau','outerV'], compiler = 'gcc' )		
		newTau = t
		
		# not needed?
		#U[u] = TU
		#V[tempMovieID[:length]] = TV
		
		# saves memory by forcing garbage collection?
		#del outerU
		#del covU
		#del incS

	U = mater(U)
	V = mater(V)
	
	for r in xrange(rank):
		newSigma[r] = (newSigma[r] + (U[:,r].T * U[:,r])[0,0])/float(numUsers-1)
	
	for i in xrange(numMovies):
		V[i] = V[i] / tau
		covPV[i] = inverter(covPV[i])
		V[i] = (covPV[i] * V[i].T).T

	#was testing this
	#for r in xrange(rank):
	#	rho[r] = 0.
	#	for m in xrange(numMovies):
	#		rho[r] += S[m,r,r]
	#	rho[r] = (rho[r] + (V[:,r].T * V[:,r])[0,0])/float(numMovies-1)
		
	# transform U and V
	#norms = numpy.diag(V.T * V)
	#for c in xrange(norms.shape[0]):
	#	V[:,c] /= math.sqrt( norms[c] * rank )
	#	U[:,c] *= math.sqrt( norms[c] * rank )	
	#del norms
	
	newTau = newTau / float(entries-1)

	del sigma
	del tau
	#compute sigma and tau
	sigma = newSigma
	tau = newTau
	
	print "tau:", tau
	print "sigma:", sigma
	#print "rho:",rho
	
	# saving
	if numUsers == 480189:
		VBFilename = "VB-"
		
		if trainingMode == 'Residual':
			VBFilename += "residual-"
		
		VBFilename += "iter-" + repr(iter) + "-rank-" + repr(int(rank)) + ".dat"
		
		VBFile = open( VBFilename, "wb" )
		VBFile.write( struct.pack('>III', numUsers,numMovies,rank) )
		VBFile.write( struct.pack('>d', tau ) )
		#for r in xrange(rank):
		#	VBFile.write( struct.pack('d', sigma[r]) )
		VBFile.write( struct.pack('>'+'d' * rank, *sigma) )
		
		for u in xrange(numUsers):
			#for r in xrange(rank):
			#	VBFile.write( struct.pack('d', U[u,r]) )			
			VBFile.write( struct.pack('>'+'d'*rank, *(U[u].T)) )			
		
		for m in xrange(numMovies):
			#for r in xrange(rank):
			#	VBFile.write( struct.pack('d', V[m,r]) )
			VBFile.write( struct.pack('>'+'d'*rank, *(V[m].T)) )
	
		#for m in xrange(numMovies):
		#	for r in xrange(rank):
		#		#for k in xrange(rank):
		#		#	VBFile.write( struct.pack('d', covPV[m,r,k]) )
		#		VBFile.write( struct.pack('d'*rank, *(covPV[m,r].T)) )
	
		#for r in xrange(rank):
		#	VBFile.write( struct.pack('d', rho[r]) )
	
		VBFile.close()	

	if trainingMode != 'Residual':
		probeFile = open("probeRatingWithDate.dat", "rb")
		str = probeFile.read( struct.calcsize('>I') )
		e = struct.unpack('>I', str)
		RMSETotalEntries = e[0]
	
		print "probe Total number of entries:",RMSETotalEntries
	
		sumSquaredValues = 0.0
	
		numEntries = 0
		
		for i in xrange(RMSETotalEntries):
			str = probeFile.read( struct.calcsize('>IHBBBBB') )
			e = struct.unpack('>IHBBBBB',str)
		
			movieID = e[1]
			userID = e[0]
			rating = e[2]
			#dayOfWeek = e[3]
			#week = e[4]
			#year = e[5]
			#holiday = e[6]
		
			if userID >= numUsers: continue

			numEntries = numEntries + 1
			
			# make prediction
			pred = (U[userID,:]*V[movieID,:].T)[0,0]
				
			pred = max( pred, 1.0 )
			pred = min( pred, 5.0 )
			
			delta = float(rating) - pred
		
			sumSquaredValues = sumSquaredValues + delta * delta 
		
		print repr(numEntries) + " pairs RMSE: " + repr(math.sqrt(sumSquaredValues/float(numEntries)))

	# saves memory by forcing garbage collection?
	#del PV
	#del U
	
	PV[:] = V
