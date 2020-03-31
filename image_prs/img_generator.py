from PIL import Image

# Load Colon Cancer data and generate images

betaFile = open("emerge-prs-risk-beta.csv", "r")
betaFile = [x.split(",") for x in betaFile.readlines()]
betaHeader = betaFile[0]
betaFile = betaFile[1:]

for snp in betaFile:
    snp[-1] = snp[-1].split("\n")[0]

variantFile = open("concatenated_merged_variants.raw", "r")
variantFile = [x.split(" ") for x in variantFile.readlines()]
variantHeader = variantFile[0]
variantHeader[-1] = variantHeader[-1].split("\n")[0]
variantFile = variantFile[1:]

for ind in variantFile:
    ind[-1] = ind[-1].split("\n")[0]

betaDict = dict()
for snp in betaFile:
    snpName = snp[2].split("_")[0]
    betaDict[snpName] = snp

print(len(variantHeader[6:]))

count = 6
headDict = dict()
for var in variantHeader[6:]:
    varName = var.split("_")[0]
    varValue = var.split("_")[1]
    headDict[varName] = [count, varValue]
    count = count + 1

indDict = dict()
for ind in variantFile:
    indName = ind[1]
    indList = list()
    for snp in betaDict.keys():
        riskSNP = betaDict[snp][4] if betaDict[snp][6] == '1' else betaDict[snp][5]
        colNum = headDict[snp][0]
        colVal = headDict[snp][1]
        indVal = int(ind[int(colNum)]) if riskSNP == colVal else (2 - int(ind[int(colNum)]))
        indList.append((snp, indVal))
    indDict[indName] = indList

caseStatusFile = open("emerge-crc-status-20190-06-06.txt", "r")
caseStatusFile = [x.split("\t") for x in caseStatusFile.readlines()]
caseStatusHeader = caseStatusFile[0]
caseStatusFile = caseStatusFile[1:]

for ind in caseStatusFile:
    ind[-1] = ind[-1].split("\n")[0]

statusDict = dict()

for ind in caseStatusFile:
    statusDict[ind[0]] = int(ind[4])

for ind in indDict.keys():
    new = Image.new("RGB", (1, 140), "white")
    pixels = new.load()
    indValue = indDict[ind]
    print(ind, indValue)
    for i in range(1):
        for j in range(140):
            snp = indValue[j]
            snpCount = snp[1]
            pixels[i,j] = ((2 - int(snpCount)) * 127, (2 - int(snpCount)) * 127, (2 - int(snpCount)) * 127)
    if ind in statusDict.keys():
        if statusDict[ind] == 0:
            new.save("images/controls/" + ind + ".png", "png")
        else:
            new.save("images/cases/" + ind + ".png", "png")





