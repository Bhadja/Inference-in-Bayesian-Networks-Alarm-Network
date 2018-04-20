import sys,copy,random

#----------------------variable list----------------------------#
bayesnet = dict()
eviVector = dict()
queryEviVector = dict()
Accumulator = dict()
evidence = sys.argv[1].replace("[","").replace("]","").split(">,<")
evidence = [evi.replace("<","").replace(">","").split(",") for evi in evidence]
query = sys.argv[2].replace("[","").replace("]","").split(",")
eviKeys = [e[0] for e in evidence]
queryKeys =  [q for q in query]
queryEviKeys = queryKeys + eviKeys
sampleSizes = (10,50,100,200,500,1000,10000)

# predefined bayesnet
def network():
	parent = list()

	parenta = list()
	parenta.append("b")
	parenta.append("e")

	parentJM = list()
	parentJM.append("a")

	bayesnet.update({"b": (parent, {"t": 0.001})  })

	bayesnet.update({"e": (parent, {"t": 0.002})})

	bayesnet.update({"a": (parenta, {"tt": 0.95, "tf": 0.94, "ft": 0.29, "ff": 0.001})})

	bayesnet.update({"j": (parentJM, {"t": 0.90, "f": 0.05})})

	bayesnet.update({"m": (parentJM, {"t": 0.70, "f": 0.01})})
	# print(cpt)

network()

for e in bayesnet.keys():
	eviVector[e] = -1

for e in evidence:
	if(e[1]=="t"):
		eviVector[e[0]] = 1
	else:
		eviVector[e[0]] = 0

queryEviVector = copy.copy(eviVector)
for q in query:
	queryEviVector[q] = 1


def getParentValue(sample,e):
	parentValue = ""
	parent = bayesnet.get(e)[0]
	if not parent:
		return "t"
	else:
		for p in parent:
			if sample.get(p) == 1:
				parentValue += "t"
			else:
				parentValue += "f"

	return parentValue


def generateSamples(sampleSize,algoType):
	Sample = list()

	for i in range(sampleSize):
		rejected = False
		sample = dict()
		if algoType == "lw":
			weightVal = 1.00

		for e in bayesnet.keys():
			sample[e] = -1

		for e in bayesnet.keys():
			ranNum = random.random()
			parentValue =  getParentValue(sample,e)
			cptEntry = bayesnet.get(e)[1].get(parentValue)

			if(ranNum <= cptEntry):
				if algoType == "r" and eviVector[e] == 0:
					rejected = True
					break
				else:
					sample[e] = 1

				if algoType == "lw" and e in eviKeys:
					weightVal = weightVal*cptEntry

			else:
				if algoType == "r" and eviVector[e] == 1:
					rejected = True
					break
				else:
					sample[e] = 0

				if algoType == "lw" and e in eviKeys:
					weightVal = weightVal*(1-cptEntry)

		if rejected:
			continue
		if(algoType == "lw"):
			Sample.append((sample,weightVal))
		else:
			Sample.append(sample)

	return Sample

# True if not same (reverse from natural interpretaion)
def checkNotSimilarity(sample, vector):
	flag = False
	for e in bayesnet.keys():
		if sample[e] != vector[e] and vector[e] != -1:
			flag = True

	return flag

def getProbablity(Samples, algoType):
	eviSample = 0.00
	eviQueSample = 0.00
	for sampleW in Samples:
		if algoType =="lw":
			sample = sampleW[0]
			weight = sampleW[1]
		else:
			sample = sampleW

		if not checkNotSimilarity(sample,queryEviVector):
			if algoType == "lw":
				eviQueSample = eviQueSample + weight
			else:
				eviQueSample = eviQueSample + 1.0


		if not checkNotSimilarity(sample,eviVector):
			if algoType == "lw":
				eviSample = eviSample + weight
			else:
				eviSample = eviSample + 1.0


	if eviSample != 0:
		return eviQueSample/eviSample
	else:
		return 0.00

#----------------------------Sampling Algorithm Entry Point------------------------------#
print("|"+ "Prior Sampling" +"|"+ "\t\t" +\
     "|" + "Rejection Sampling"      +"|" + "\t\t" +\
      "|" +"Likelihood Weighting" +"|")
for sampleSize in sampleSizes:
	addP = inferedProbP = 0.00
	addR = inferedProbR = 0.00
	addLW = inferedProbLW = 0.00

	for i in range(10):

		priorSample = generateSamples(sampleSize,"p")
		inferedProbP = getProbablity(priorSample,"p")
		addP = addP + inferedProbP

		rejectionSample = generateSamples(sampleSize,"r")
		inferedProbR = getProbablity(rejectionSample,"r")
		addR = addR + inferedProbR

		LWSample = generateSamples(sampleSize, "lw")
		inferedProbLW = getProbablity(LWSample, "lw")
		addLW = addLW + inferedProbLW/10.0


	Accumulator["p"] = addP/10.00
	Accumulator["r"] = addR/10.00
	Accumulator["lw"] = addLW/10.00
	print("|"+ "{}".format(round(Accumulator["p"], 9))+ "|" + "\t\t" +\
     "|" + "{}".format(round(Accumulator["r"], 9))+ "|"  + "\t\t" +\
      "|" +"{}".format(round(Accumulator["lw"], 9)) +"|")

#--------------------Enumeration Algo Starts here-----------------------------#
varss = ['m','j','a','b','e']

def Pr(var, val, e, bayesnet):
    parents = bayesnet[var][0]
    if len(parents) == 0:
        Pr = bayesnet[var][1]["t"]
    else:
        parentVals = "".join([e[parent] for parent in parents])
        Pr = bayesnet[var][1][parentVals]
    if val=="t": return Pr
    else: return 1.0-Pr

def enumerationAsk(X, e, bayesnet,varss):
    QX = {}
    for xi in ["f","t"]:
        e[X] = xi
        QX[xi] = enumerateAll(varss,e,bayesnet)
        del e[X]
    Qx = normalize(QX)
    return Qx["t"]


def enumerateAll(varss, e,bayesnet):
    if len(varss) == 0: return 1.0
    Y = varss.pop()
    if Y in e:
        val = Pr(Y,e[Y],e,bayesnet) * enumerateAll(varss,e,bayesnet)
        varss.append(Y)
        return val
    else:
        total = 0
        e[Y] = "t"
        total += Pr(Y,"t",e,bayesnet) * enumerateAll(varss,e,bayesnet)
        e[Y] = "f"
        total += Pr(Y,"f",e,bayesnet) * enumerateAll(varss,e,bayesnet)
        del e[Y]
        varss.append(Y)
        return total

def normalize(QX):
    total = 0.0
    for val in QX.values():
        total += val
    for key in QX.keys():
        QX[key] /= total
    return QX

# print()
exactInference = 0.00
for q in query:
	exactInference = enumerationAsk(q,{k[0]: str(k[1]) for k in evidence},bayesnet,varss)
	print(str("Exact Inference")+" "+str(exactInference))

