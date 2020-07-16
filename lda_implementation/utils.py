
from pyvi import ViTokenizer, ViPosTagger
from pyvi import ViUtils

stop_words = []
stop_words_file = "./VNmese-stopwords.txt"
stop_words_no_accent = []
with open(stop_words_file, 'r') as f:
    line = f.readline()
    while line:
        line = ViTokenizer.tokenize(u""+line)
        stop_words.append(line)
        stop_words_no_accent.append(ViUtils.remove_accents(u""+line))
        line = f.readline()

def remove_stop_words(line):
    # Remove stopwords using binary search
    # So freaking cồng kềnh, I know
    # But screw it, lol
    words = line.split()
    mid = 0
    line = ""
    for count,word in enumerate(words):
        word_no_accent = ViUtils.remove_accents(u""+word)
        left = 0
        right = len(stop_words)-1
        while(int((right-left)/2)>0):
            mid = int((right+left)/2)
            if(word_no_accent>stop_words_no_accent[mid]):
                left = mid
            elif(word_no_accent<stop_words_no_accent[mid]):
                right = mid
            elif((word_no_accent==stop_words_no_accent[mid])):
                break
        if(word_no_accent != stop_words_no_accent[mid]):
            line = line+word+" "
        else:
            if(mid>=3 and mid<=(len(stop_words)-4)):
                check = 0
                for i in range(mid-3,mid+3):
                    if(word==stop_words[i]):
                        check = 1
                        break
                if(check == 0):
                    line = line+word+" "
            elif(mid <=3):
                check = 0
                for i in range(0,mid+3):
                    if(word==stop_words[i]):
                        check = 1
                        break
                if(check == 0):
                    line = line+word+" "
            elif(mid >= (len(stop_words)-4)):
                check = 0
                for i in range(mid - 5,mid):
                    if(word==stop_words[i]):
                        check = 1
                        break
                if(check == 0):
                    line = line+word+" "
    return line