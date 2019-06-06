text = "Ha - turns out my major problem was in forgetting to remind python to use floats when dividing! Thanks for your help though guys"
text = "Encoding as a type that natively iterates by ordinal means the conversion goes much faster; in local tests on both Py2.7 and Py3.5, iterating a str to get its ASCII codes using map(ord, mystr) starts off taking about twice as long for a len 10 str than using bytearray(mystr) on Py2 or "

text2 = "gibberish ads  dnlwe wae ew af sdlgoew le wds af"
filename = "OnTheOriginOfSpecies.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

import sys
import collections

def ic(text):
    # remove all non alpha and whitespace and force uppercase
    flattext = "".join([x.upper() for x in text.split() if x.isalpha()])
    N = len(flattext)
    freqs = collections.Counter(flattext)
    alphabet = map(chr, range(ord('A'), ord('Z')+1))
    freqsum = 0.0

    # math
    for letter in alphabet:
        freqsum += freqs[letter] * (freqs[letter] - 1)

    IC = freqsum / (N*(N-1))

    return IC

gen = "the of thd en thei tie eerts to theit sa thgs soe oese nhd eoudit tf thdia bn toe sf bert tn the oes b ohran meres a  ar tay oh ra cj cooher de ho sodol ae cenere to co thac io oot eese bnefe ohdtari andsenhn of tieh rhe of tee se nndgr benaevor bn a sar  she sa th ttes rn thyuo th the of the a saralt oe tht srrg gr moyel ar io the cned cr tht seatd th the cu to the cne oo thea oh thecr se the  at ia toc st satts sh th c gona ai the oaln ar nery neeo bend thl eo th thee the se thel errmited vheeo oe thest in theecds seeheeiut ranu l ggsg b shr of the ae d vhel be ta c tienln sfn shdes be ta then the of mh thrhen ao an tuhsn drru an fyeuna oe thels of they the si thetho rn the sh the shrho te th the ao thr shes th the lfdsa sfe ennms oh the vor oe ae ra thel th theore oo the cort anry lost ohe sh thr sf thd soe eni en the calue an ifsds oo the me the oe theh tasiec ano thet piy shscre brdmd bastel sheceeoa whetessrdn vf the oa thdl thna pheetis oo the toect ieret oer ee tf the ff thd"

print(ic(text))
print(ic(text2))
print(ic(raw_text))
print(ic(gen))

